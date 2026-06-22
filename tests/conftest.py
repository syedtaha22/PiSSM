"""
Shared test fixtures for integration tests.

Provides an in-process gRPC server with the NodeServiceServicer
registered, and a channel connected to it. No real network - the
server binds to localhost on an OS-assigned ephemeral port.
"""

from concurrent import futures

import grpc
import pytest

from proto.generated import nodes_pb2_grpc
from orchestrator.node_registry import NodeRegistry
from orchestrator.service import NodeServiceServicer


@pytest.fixture
def registry():
    """
    Return a fresh NodeRegistry instance.
    """
    return NodeRegistry()


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
