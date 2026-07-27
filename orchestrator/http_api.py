"""
FastAPI HTTP API for the orchestrator.

Exposes the node registry and model registry over HTTP so the TUI
and WebUI can both integrate against a single interface, without
speaking gRPC directly. Runs inside the same orchestrator process as
the gRPC server, sharing the same NodeRegistry and ModelRegistry
instances.
"""

from dataclasses import asdict

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference.manifest import ManifestError, ModelManifest, manifest_from_dict
from inference.model_registry import ModelRegistry
from orchestrator.node_registry import NodeInfo, NodeRegistry


class ModelSubmission(BaseModel):
    """
    Request body for submitting a model manifest.

    Parameters
    ----------
    manifest_yaml : str
        Raw YAML content of the model manifest.
    """

    manifest_yaml: str


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


def create_app(registry: NodeRegistry, model_registry: ModelRegistry) -> FastAPI:
    """
    Build the orchestrator's FastAPI application.

    Parameters
    ----------
    registry : NodeRegistry
        The live node registry to expose via GET /nodes.
    model_registry : ModelRegistry
        The model registry to expose via GET /models and POST /models.

    Returns
    -------
    FastAPI
        The configured application, ready to run under uvicorn or a
        test client.
    """
    app = FastAPI(title="PiSSM Orchestrator API")

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

    return app
