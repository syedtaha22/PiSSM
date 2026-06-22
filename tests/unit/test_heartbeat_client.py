"""
Tests for the HeartbeatClient.

Validates that the client sends heartbeats to a real in-process gRPC
server, stops cleanly, and handles server unavailability without
crashing. Each test creates its own gRPC server with a heartbeat
interval matching the client's, so the server does not override
the client's test interval.
"""

import time
from concurrent import futures

import grpc

from proto.generated import nodes_pb2_grpc
from orchestrator.node_registry import NodeRegistry
from orchestrator.service import NodeServiceServicer
from worker.heartbeat import HeartbeatClient

TEST_INTERVAL_S = 0.2
TEST_INTERVAL_MS = int(TEST_INTERVAL_S * 1000)


def start_test_server(heartbeat_interval_ms=TEST_INTERVAL_MS):
    """
    Start a gRPC server with a NodeServiceServicer configured for testing.

    Parameters
    ----------
    heartbeat_interval_ms : int
        The heartbeat interval the servicer returns in responses.

    Returns
    -------
    tuple[grpc.Server, str, NodeRegistry]
        The running server, the address string, and the registry.
    """
    registry = NodeRegistry()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = NodeServiceServicer(
        registry, heartbeat_interval_ms=heartbeat_interval_ms
    )
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, f"localhost:{port}", registry


class TestHeartbeatClientLifecycle:
    """
    Tests for starting and stopping the heartbeat client.
    """

    def test_client_sends_heartbeat(self):
        """
        The client sends at least one heartbeat to the server after starting.
        """
        server, address, registry = start_test_server()

        client = HeartbeatClient(
            orchestrator_address=address,
            node_id="test-node",
            interval_s=TEST_INTERVAL_S,
        )
        client.start()
        time.sleep(0.5)
        client.stop()
        server.stop(grace=0)

        node = registry.get_node("test-node")
        assert node is not None
        assert node.node_id == "test-node"
        assert node.status == "available"

    def test_client_stops_cleanly(self):
        """
        After stop(), is_running is False.
        """
        server, address, _ = start_test_server()

        client = HeartbeatClient(
            orchestrator_address=address,
            node_id="test-node",
            interval_s=TEST_INTERVAL_S,
        )
        client.start()
        assert client.is_running is True

        client.stop()
        server.stop(grace=0)
        assert client.is_running is False

    def test_client_sends_multiple_heartbeats(self):
        """
        After running for several intervals, the most recent heartbeat
        timestamp is close to the current time, proving heartbeats
        continued periodically and not just on startup.
        """
        server, address, registry = start_test_server()

        client = HeartbeatClient(
            orchestrator_address=address,
            node_id="test-node",
            interval_s=TEST_INTERVAL_S,
        )
        client.start()
        time.sleep(1.0)
        last_hb = registry.get_node("test-node").last_heartbeat
        now = time.monotonic()
        client.stop()
        server.stop(grace=0)

        assert now - last_hb < 0.5


class TestHeartbeatClientResilience:
    """
    Tests for error handling when the server is unavailable.
    """

    def test_client_handles_server_unavailable(self):
        """
        The client does not crash when the orchestrator is unreachable.
        It logs a warning and retries on the next interval.
        """
        client = HeartbeatClient(
            orchestrator_address="localhost:1",
            node_id="test-node",
            interval_s=TEST_INTERVAL_S,
        )
        client.start()
        time.sleep(0.5)
        client.stop()

        assert client.is_running is False
