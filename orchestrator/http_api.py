"""
FastAPI HTTP API for the orchestrator.

Exposes the node registry, model registry, and pipeline inference
over HTTP so the TUI and WebUI can both integrate against a single
interface, without speaking gRPC directly. Runs inside the same
orchestrator process as the gRPC server, sharing the same
NodeRegistry, ModelRegistry, and ResultStore instances.
"""

import threading
import time
from dataclasses import asdict, dataclass

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from transformers import AutoTokenizer

from inference.manifest import ManifestError, ModelManifest, manifest_from_dict
from inference.model_registry import ModelRegistry
from orchestrator.dispatch import DispatchError, plan_dispatch
from orchestrator.model_store import ModelStore
from orchestrator.node_registry import NodeInfo, NodeRegistry
from orchestrator.pipeline import PipelineRunner, ResultStore

DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_INFER_TIMEOUT_S = 120.0


class ModelSubmission(BaseModel):
    """
    Request body for submitting a model manifest.

    Parameters
    ----------
    manifest_yaml : str
        Raw YAML content of the model manifest.
    """

    manifest_yaml: str


class InferSubmission(BaseModel):
    """
    Request body for running inference on a registered model.

    Parameters
    ----------
    model_name : str
        Name of a previously registered model.
    input : str
        Input text prompt.
    max_new_tokens : int
        Number of tokens to generate.
    """

    model_name: str
    input: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


@dataclass
class _InferenceSession:
    """
    A resident, loaded pipeline for one model, kept across requests.

    Parameters
    ----------
    runner : PipelineRunner
        The loaded pipeline runner for this model.
    model_store : ModelStore
        The orchestrator-side full-model host used to build the runner.
    tokenizer : transformers.PreTrainedTokenizerBase
        The tokenizer for this model.
    num_nodes : int
        Number of worker nodes the model is dispatched across.
    """

    runner: PipelineRunner
    model_store: ModelStore
    tokenizer: object
    num_nodes: int


def _node_to_dict(node: NodeInfo) -> dict:
    """
    Convert a NodeInfo into a JSON-serializable dict.

    Parameters
    ----------
    node : NodeInfo
        The node snapshot to convert.

    Returns
    -------
    dict
        All node fields, keyed by field name.
    """
    return asdict(node)


def _manifest_to_dict(manifest: ModelManifest) -> dict:
    """
    Convert a ModelManifest into a JSON-serializable dict.

    Parameters
    ----------
    manifest : ModelManifest
        The manifest to convert.

    Returns
    -------
    dict
        All manifest fields, keyed by field name.
    """
    return asdict(manifest)


def create_app(
    registry: NodeRegistry,
    model_registry: ModelRegistry,
    result_store: ResultStore,
    callback_address: str,
) -> FastAPI:
    """
    Build the orchestrator's FastAPI application.

    Parameters
    ----------
    registry : NodeRegistry
        The live node registry to expose via GET /nodes, and to plan
        dispatch against for POST /infer.
    model_registry : ModelRegistry
        The model registry to expose via GET /models and POST /models.
    result_store : ResultStore
        The pipeline result store shared with the orchestrator's
        PipelineCallbackServicer, used to wait for POST /infer results.
    callback_address : str
        Address ("host:port") of the orchestrator's
        PipelineCallbackService, passed to every worker so the last
        shard in a pipeline knows where to deliver its result.

    Returns
    -------
    FastAPI
        The configured application, ready to run under uvicorn or a
        test client.
    """
    app = FastAPI(title="PiSSM Orchestrator API")
    sessions: dict[str, _InferenceSession] = {}
    sessions_lock = threading.Lock()

    def _get_or_create_session(manifest: ModelManifest) -> _InferenceSession:
        """
        Return the resident session for a model, creating it if needed.

        Raises
        ------
        HTTPException
            503 if no worker nodes are available, 400 for other
            dispatch errors.
        """
        with sessions_lock:
            session = sessions.get(manifest.name)
            if session is not None:
                return session

            try:
                plan = plan_dispatch(manifest, registry)
            except DispatchError as err:
                if "no available" in str(err):
                    raise HTTPException(status_code=503, detail=str(err))
                raise HTTPException(status_code=400, detail=str(err))

            model_store = ModelStore()
            model_store.load(manifest)

            runner = PipelineRunner(
                model_store=model_store,
                plan=plan,
                orchestrator_callback_address=callback_address,
                result_store=result_store,
                timeout_s=DEFAULT_INFER_TIMEOUT_S,
            )
            runner.load()

            tokenizer = AutoTokenizer.from_pretrained(manifest.tokenizer)

            session = _InferenceSession(
                runner=runner,
                model_store=model_store,
                tokenizer=tokenizer,
                num_nodes=len(plan.assignments),
            )
            sessions[manifest.name] = session
            return session

    def _run_inference(submission: InferSubmission) -> dict:
        """
        Run generation for one request against a resident pipeline session.
        """
        manifest = model_registry.get(submission.model_name)
        if manifest is None:
            raise HTTPException(
                status_code=404, detail=f"Model '{submission.model_name}' not found"
            )

        session = _get_or_create_session(manifest)

        input_ids = session.tokenizer(submission.input, return_tensors="pt").input_ids

        start = time.monotonic()
        result = None
        for _ in range(submission.max_new_tokens):
            result = session.runner.run_forward(input_ids)
            next_token = (
                result.output_tensor[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
        latency_ms = (time.monotonic() - start) * 1000

        output_text = session.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "output": output_text,
            "latency_ms": latency_ms,
            "node_latencies_ms": list(result.node_latencies_ms) if result else [],
            "peak_memory_mb": list(result.node_peak_memory_mb) if result else [],
            "num_nodes": session.num_nodes,
        }

    @app.get("/nodes")
    def get_nodes() -> list[dict]:
        """
        List all known worker nodes with their current status.
        """
        return [_node_to_dict(node) for node in registry.list_nodes()]

    @app.get("/models")
    def get_models() -> list[dict]:
        """
        List all registered model manifests.
        """
        return [_manifest_to_dict(m) for m in model_registry.list_models()]

    @app.post("/models", status_code=201)
    def submit_model(submission: ModelSubmission) -> dict:
        """
        Validate and register a model manifest.

        Raises
        ------
        HTTPException
            400 if the YAML is malformed or the manifest is invalid,
            409 if a model with the same name is already registered.
        """
        try:
            data = yaml.safe_load(submission.manifest_yaml)
        except yaml.YAMLError as err:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {err}")

        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400, detail="Manifest must be a YAML mapping"
            )

        try:
            manifest = manifest_from_dict(data)
        except ManifestError as err:
            raise HTTPException(status_code=400, detail=str(err))

        try:
            model_registry.register(manifest)
        except ValueError as err:
            raise HTTPException(status_code=409, detail=str(err))

        return _manifest_to_dict(manifest)

    @app.post("/infer")
    async def infer(submission: InferSubmission) -> dict:
        """
        Run inference on a registered model.

        Dispatches and loads the model's pipeline across available
        worker nodes on first use, then reuses the resident pipeline
        for subsequent requests. Runs the blocking pipeline calls in a
        thread pool so the event loop stays free to serve other routes.

        Raises
        ------
        HTTPException
            404 if the model is not registered, 503 if no worker nodes
            are available, 400 for other dispatch errors.
        """
        return await run_in_threadpool(_run_inference, submission)

    return app
