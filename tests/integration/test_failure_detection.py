"""
Integration tests for failure detection and auto-rejoin.

These tests run a real orchestrator (gRPC server + reaper thread) and
real HeartbeatClients in-process. They verify the full lifecycle:
nodes register, a node drops and is reaped, a node rejoins.
Short intervals (0.1s heartbeat, 0.3s timeout) keep tests fast.
"""

import time
from concurrent import futures

import grpc

from proto.generated import nodes_pb2_grpc
from orchestrator.node_registry import NodeRegistry
from orchestrator.service import NodeServiceServicer
from orchestrator.server import run_reaper
from worker.heartbeat import HeartbeatClient

import threading

HEARTBEAT_INTERVAL_S = 0.1
MISSED_THRESHOLD = 3
REAPER_INTERVAL_S = 0.05
HEARTBEAT_INTERVAL_MS = int(HEARTBEAT_INTERVAL_S * 1000)


def start_orchestrator():
    """
    Start a full orchestrator: gRPC server + reaper thread.

    Returns
    -------
    tuple[grpc.Server, NodeRegistry, threading.Event]
        The running server, the registry, and the reaper stop event.
    """
    registry = NodeRegistry(
        heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
        missed_threshold=MISSED_THRESHOLD,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = NodeServiceServicer(
        registry,
        heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
    )
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    stop_event = threading.Event()
    reaper_thread = threading.Thread(
        target=run_reaper,
        args=(registry, REAPER_INTERVAL_S, stop_event),
        daemon=True,
    )
    reaper_thread.start()

    return server, f"localhost:{port}", registry, stop_event


def stop_orchestrator(server, stop_event):
    """
    Cleanly shut down the orchestrator server and reaper thread.

    Parameters
    ----------
    server : grpc.Server
        The gRPC server to stop.
    stop_event : threading.Event
        The event that signals the reaper thread to exit.
    """
    stop_event.set()
    server.stop(grace=0)


class TestFailureDetection:
    """
    Tests for the reaper marking nodes as unavailable after missed heartbeats.
    """

    def test_node_drops_after_missed_heartbeats(self):
        """
        A node that stops sending heartbeats is marked unavailable
        within the timeout window.
        """
        server, address, registry, stop_event = start_orchestrator()

        client = HeartbeatClient(
            orchestrator_address=address,
            node_id="node-1",
            interval_s=HEARTBEAT_INTERVAL_S,
        )
        client.start()
        time.sleep(0.3)
        assert registry.get_node("node-1").status == "available"

        client.stop()
        timeout = HEARTBEAT_INTERVAL_S * MISSED_THRESHOLD + REAPER_INTERVAL_S + 0.2
        time.sleep(timeout)

        assert registry.get_node("node-1").status == "unavailable"

        stop_orchestrator(server, stop_event)

    def test_node_rejoins_after_restart(self):
        """
        A node marked unavailable is restored to available when it
        starts sending heartbeats again.
        """
        server, address, registry, stop_event = start_orchestrator()

        client = HeartbeatClient(
            orchestrator_address=address,
            node_id="node-1",
            interval_s=HEARTBEAT_INTERVAL_S,
        )
        client.start()
        time.sleep(0.3)
        client.stop()

        timeout = HEARTBEAT_INTERVAL_S * MISSED_THRESHOLD + REAPER_INTERVAL_S + 0.2
        time.sleep(timeout)
        assert registry.get_node("node-1").status == "unavailable"

        client2 = HeartbeatClient(
            orchestrator_address=address,
            node_id="node-1",
            interval_s=HEARTBEAT_INTERVAL_S,
        )
        client2.start()
        time.sleep(0.3)

        assert registry.get_node("node-1").status == "available"

        client2.stop()
        stop_orchestrator(server, stop_event)


class TestEndToEnd:
    """
    Sprint acceptance test: two workers register, one dies, the other stays.
    """

    def test_two_workers_one_dies(self):
        """
        Two workers register with the orchestrator. One is killed.
        The dead worker is marked unavailable while the live worker
        remains available.
        """
        server, address, registry, stop_event = start_orchestrator()

        client1 = HeartbeatClient(
            orchestrator_address=address,
            node_id="worker-1",
            interval_s=HEARTBEAT_INTERVAL_S,
        )
        client2 = HeartbeatClient(
            orchestrator_address=address,
            node_id="worker-2",
            interval_s=HEARTBEAT_INTERVAL_S,
        )

        client1.start()
        client2.start()
        time.sleep(0.3)

        assert len(registry.list_nodes(status_filter="available")) == 2

        client2.stop()
        timeout = HEARTBEAT_INTERVAL_S * MISSED_THRESHOLD + REAPER_INTERVAL_S + 0.2
        time.sleep(timeout)

        available = registry.list_nodes(status_filter="available")
        unavailable = registry.list_nodes(status_filter="unavailable")

        assert len(available) == 1
        assert available[0].node_id == "worker-1"
        assert len(unavailable) == 1
        assert unavailable[0].node_id == "worker-2"

        client1.stop()
        stop_orchestrator(server, stop_event)
