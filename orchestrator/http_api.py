"""
FastAPI HTTP API for the orchestrator.

Exposes the node registry, model registry, pipeline inference, and
the live dispatch topology over HTTP so the TUI and WebUI can both
integrate against a single interface, without speaking gRPC
directly. Runs inside the same orchestrator process as the gRPC
server, sharing the same NodeRegistry, ModelRegistry, and ResultStore
instances.

Model loading is decoupled from inference: POST /models/{name}/load
kicks off dispatch and shard loading in a background thread, and
GET /models/{name}/status reports progress. POST /infer still loads
a model on demand if it hasn't been explicitly preloaded, so the
endpoint remains usable standalone.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer

from inference.manifest import ManifestError, ModelManifest, manifest_from_dict
from inference.model_registry import ModelRegistry
from orchestrator.dispatch import DispatchError, DispatchPlan, plan_dispatch
from orchestrator.model_store import ModelStore
from orchestrator.node_registry import NodeInfo, NodeRegistry
from orchestrator.pipeline import PipelineRunner, ResultStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_INFER_TIMEOUT_S = 120.0
DEFAULT_LOAD_WAIT_TIMEOUT_S = 300.0
STATUS_POLL_INTERVAL_S = 0.05


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
    """

    runner: PipelineRunner
    model_store: ModelStore
    tokenizer: object


@dataclass
class _ModelState:
    """
    Tracks a model's dispatch/load progress, keyed by model name.

    Parameters
    ----------
    status : str
        One of "loading", "ready", "error".
    plan : DispatchPlan or None
        The dispatch plan, known as soon as dispatch succeeds - before
        the (slow) shard loading step completes. None only if dispatch
        itself failed.
    session : _InferenceSession or None
        The loaded pipeline, set once status is "ready".
    error : str or None
        Error message, set once status is "error".
    """

    status: str
    plan: DispatchPlan | None = None
    session: _InferenceSession | None = None
    error: str | None = None


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
    # The WebUI is served same-origin in production (mounted below), but
    # during development `next dev` runs on its own port and calls this
    # API cross-origin. Permissive CORS is fine for a LAN-only cluster tool.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    model_states: dict[str, _ModelState] = {}
    states_lock = threading.Lock()

    def _build_session(
        manifest: ModelManifest, plan: DispatchPlan
    ) -> _InferenceSession:
        """
        Do the actual (slow) shard-loading work for an already-dispatched plan.
        """
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
        # The full model only exists to extract shard bytes for workers;
        # once every shard has been sent, the orchestrator's own copy is
        # dead weight and can be freed.
        model_store.unload()

        tokenizer = AutoTokenizer.from_pretrained(manifest.tokenizer)

        return _InferenceSession(
            runner=runner, model_store=model_store, tokenizer=tokenizer
        )

    def _ensure_loading_started(manifest: ModelManifest) -> _ModelState:
        """
        Return the model's current state, starting a background load if
        one hasn't already been started.
        """
        with states_lock:
            state = model_states.get(manifest.name)
            if state is not None and state.status in ("loading", "ready"):
                return state

        try:
            plan = plan_dispatch(manifest, registry)
        except DispatchError as err:
            state = _ModelState(status="error", error=str(err))
            with states_lock:
                model_states[manifest.name] = state
            return state

        state = _ModelState(status="loading", plan=plan)
        with states_lock:
            model_states[manifest.name] = state

        def _load() -> None:
            try:
                session = _build_session(manifest, plan)
                with states_lock:
                    model_states[manifest.name] = _ModelState(
                        status="ready", plan=plan, session=session
                    )
            except Exception as err:
                logger.exception("Failed to load model '%s'", manifest.name)
                with states_lock:
                    model_states[manifest.name] = _ModelState(
                        status="error", plan=plan, error=str(err)
                    )

        threading.Thread(target=_load, daemon=True).start()
        return state

    def _get_ready_session(
        manifest: ModelManifest, wait_timeout_s: float = DEFAULT_LOAD_WAIT_TIMEOUT_S
    ) -> _InferenceSession:
        """
        Block until the model is loaded, starting a load if needed.

        Raises
        ------
        HTTPException
            503 if no worker nodes are available, 400 for other dispatch
            errors, 504 if loading doesn't finish before wait_timeout_s.
        """
        _ensure_loading_started(manifest)
        deadline = time.monotonic() + wait_timeout_s
        while True:
            with states_lock:
                state = model_states[manifest.name]
            if state.status == "ready":
                return state.session
            if state.status == "error":
                if state.error and "no available" in state.error:
                    raise HTTPException(status_code=503, detail=state.error)
                raise HTTPException(status_code=400, detail=state.error)
            if time.monotonic() > deadline:
                raise HTTPException(
                    status_code=504,
                    detail=f"Timed out waiting for model '{manifest.name}' to load",
                )
            time.sleep(STATUS_POLL_INTERVAL_S)

    def _run_inference(submission: InferSubmission) -> dict:
        """
        Run generation for one request against a resident pipeline session.
        """
        manifest = model_registry.get(submission.model_name)
        if manifest is None:
            raise HTTPException(
                status_code=404, detail=f"Model '{submission.model_name}' not found"
            )

        session = _get_ready_session(manifest)

        input_ids = session.tokenizer(submission.input, return_tensors="pt").input_ids

        start = time.monotonic()
        result = session.runner.generate(input_ids, submission.max_new_tokens)
        latency_ms = (time.monotonic() - start) * 1000

        output_text = session.tokenizer.decode(
            result.output_ids[0], skip_special_tokens=True
        )
        last_step = result.step_results[-1] if result.step_results else None

        with states_lock:
            num_nodes = len(model_states[submission.model_name].plan.assignments)

        return {
            "output": output_text,
            "latency_ms": latency_ms,
            "node_latencies_ms": list(last_step.node_latencies_ms) if last_step else [],
            "peak_memory_mb": list(last_step.node_peak_memory_mb) if last_step else [],
            "num_nodes": num_nodes,
            "num_tokens": submission.max_new_tokens,
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

    @app.post("/models/{name}/load")
    def load_model(name: str) -> dict:
        """
        Start (or report on) loading a model's pipeline into worker RAM.

        Idempotent: calling this while a load is already in progress or
        complete just returns the current state without starting a
        second load.

        Raises
        ------
        HTTPException
            404 if the model is not registered.
        """
        manifest = model_registry.get(name)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        state = _ensure_loading_started(manifest)
        return {
            "status": state.status,
            "error": state.error,
            "num_nodes": len(state.plan.assignments) if state.plan else None,
        }

    @app.post("/models/{name}/redistribute")
    def redistribute_model(name: str) -> dict:
        """
        Re-dispatch a model across the currently available nodes.

        Unlike POST /models/{name}/load, this always starts a fresh
        dispatch - it is the explicit way to pick up nodes that joined
        (or dropped out of) the cluster after the model was first
        loaded, since loading never re-dispatches on its own. Frees the
        old shards from whichever nodes previously held them (best
        effort - a node that's gone is simply skipped) before
        re-dispatching against the current node registry.

        Raises
        ------
        HTTPException
            404 if the model is not registered, 400 if it has never
            been loaded (nothing to redistribute), 409 if a load or
            redistribute is already in progress.
        """
        manifest = model_registry.get(name)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        with states_lock:
            state = model_states.get(name)
            if state is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{name}' has not been loaded yet - nothing to redistribute",
                )
            if state.status == "loading":
                raise HTTPException(
                    status_code=409, detail=f"Model '{name}' is still loading"
                )
            old_session = state.session
            del model_states[name]

        if old_session is not None:
            try:
                old_session.runner.unload()
            except Exception:
                logger.warning(
                    "Failed to cleanly unload '%s' from its previous nodes "
                    "before redistributing - continuing anyway",
                    name,
                    exc_info=True,
                )

        state = _ensure_loading_started(manifest)
        return {
            "status": state.status,
            "error": state.error,
            "num_nodes": len(state.plan.assignments) if state.plan else None,
        }

    @app.get("/models/{name}/status")
    def get_model_status(name: str) -> dict:
        """
        Report a model's current load status.

        Returns
        -------
        dict
            `status` is one of "not_loaded", "loading", "ready", "error".
        """
        manifest = model_registry.get(name)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        with states_lock:
            state = model_states.get(name)

        if state is None:
            return {"status": "not_loaded", "error": None, "num_nodes": None}

        return {
            "status": state.status,
            "error": state.error,
            "num_nodes": len(state.plan.assignments) if state.plan else None,
        }

    @app.get("/topology")
    def get_topology() -> dict:
        """
        Return the dispatch plan for the currently loading/loaded model.

        Returns
        -------
        dict
            `model_name` and `assignments` (empty list if no model has
            been loaded yet). Each assignment carries `node_id`,
            `ip_address`, `layer_start`, `layer_end`, `is_first`, and
            `is_last`.
        """
        with states_lock:
            active = [
                (name, state)
                for name, state in model_states.items()
                if state.plan is not None and state.status in ("loading", "ready")
            ]

        if not active:
            return {"model_name": None, "assignments": []}

        name, state = active[0]
        return {
            "model_name": name,
            "assignments": [
                {
                    "node_id": a.node_id,
                    "ip_address": a.ip_address,
                    "layer_start": a.layer_start,
                    "layer_end": a.layer_end,
                    "is_first": a.is_first,
                    "is_last": a.is_last,
                }
                for a in state.plan.assignments
            ],
        }

    @app.post("/infer")
    async def infer(submission: InferSubmission) -> dict:
        """
        Run inference on a registered model.

        Loads the model on demand if it hasn't already been preloaded
        via POST /models/{name}/load, then reuses the resident pipeline
        for subsequent requests. Runs the blocking pipeline calls in a
        thread pool so the event loop stays free to serve other routes.

        Raises
        ------
        HTTPException
            404 if the model is not registered, 503 if no worker nodes
            are available, 400 for other dispatch errors.
        """
        return await run_in_threadpool(_run_inference, submission)

    dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard" / "out"
    if dashboard_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard"
        )
    else:
        logger.warning(
            "Dashboard build not found at %s - WebUI will not be served. "
            "Run `npm run build` in dashboard/ to generate it.",
            dashboard_dir,
        )

    return app
