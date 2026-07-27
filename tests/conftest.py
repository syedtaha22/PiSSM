"""
Shared test fixtures for integration tests.

Provides an in-process gRPC server with the NodeServiceServicer
registered, and a channel connected to it. No real network - the
server binds to localhost on an OS-assigned ephemeral port. Also
provides a FastAPI TestClient wired to real NodeRegistry and
ModelRegistry instances for HTTP API tests.
"""

from concurrent import futures

import grpc
import pytest
from fastapi.testclient import TestClient

from proto.generated import nodes_pb2_grpc
from inference.model_registry import ModelRegistry
from orchestrator.http_api import create_app
from orchestrator.node_registry import NodeRegistry
from orchestrator.pipeline import ResultStore
from orchestrator.service import NodeServiceServicer


@pytest.fixture
def registry():
    """
    Return a fresh NodeRegistry instance.
    """
    return NodeRegistry()


@pytest.fixture
def model_registry():
    """
    Return a fresh ModelRegistry instance.
    """
    return ModelRegistry()


@pytest.fixture
def result_store():
    """
    Return a fresh ResultStore instance.
    """
    return ResultStore()


@pytest.fixture
def fastapi_test_client(registry, model_registry, result_store):
    """
    Return a FastAPI TestClient wired to the registry fixtures.

    Parameters
    ----------
    registry : NodeRegistry
        The node registry instance shared between the app and the test.
    model_registry : ModelRegistry
        The model registry instance shared between the app and the test.
    result_store : ResultStore
        The pipeline result store shared between the app and the test.

    Yields
    ------
    starlette.testclient.TestClient
        A test client for the orchestrator's FastAPI app.
    """
    app = create_app(
        registry, model_registry, result_store, callback_address="localhost:50060"
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def grpc_server_and_channel(registry):
    """
    Start an in-process gRPC server with NodeServiceServicer and
    return (server, channel, registry).

    The server binds to localhost on an OS-assigned port. The channel
    connects to it. Both are torn down after the test.

    Parameters
    ----------
    registry : NodeRegistry
        The registry instance shared between server and test.

    Yields
    ------
    tuple[grpc.Server, grpc.Channel, NodeRegistry]
        The running server, a connected channel, and the registry.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = NodeServiceServicer(registry)
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")

    yield server, channel, registry

    channel.close()
    server.stop(grace=0)
