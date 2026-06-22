"""
Orchestrator gRPC server entry point.

Starts the gRPC server with the NodeServiceServicer, launches a
background reaper thread that periodically marks unresponsive nodes
as unavailable, and blocks until interrupted.
"""

import argparse
import logging
import signal
import threading
from concurrent import futures

import grpc

from proto.generated import nodes_pb2_grpc
from orchestrator.config import (
    DEFAULT_GRPC_PORT,
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    DEFAULT_REAPER_INTERVAL_S,
)
from orchestrator.node_registry import NodeRegistry
from orchestrator.service import NodeServiceServicer

logger = logging.getLogger(__name__)


def run_reaper(registry, interval_s, stop_event):
    """
    Background loop that periodically reaps stale nodes.

    Runs until ``stop_event`` is set. On each cycle, calls
    ``registry.reap_stale_nodes()`` and logs any nodes that
    were marked unavailable.

    Parameters
    ----------
    registry : NodeRegistry
        The registry to check for stale nodes.
    interval_s : float
        Seconds between reaper cycles.
    stop_event : threading.Event
        Set this event to stop the reaper loop.
    """
    while not stop_event.is_set():
        reaped = registry.reap_stale_nodes()
        for node_id in reaped:
            logger.warning("Node '%s' marked unavailable (missed heartbeats)", node_id)
        stop_event.wait(timeout=interval_s)


def create_server(
    port=DEFAULT_GRPC_PORT,
    heartbeat_interval_s=DEFAULT_HEARTBEAT_INTERVAL_S,
    missed_threshold=DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    reaper_interval_s=DEFAULT_REAPER_INTERVAL_S,
    max_workers=10,
):
    """
    Create and configure the orchestrator gRPC server.

    Parameters
    ----------
    port : int
        The port to bind the gRPC server to.
    heartbeat_interval_s : float
        Expected heartbeat interval in seconds.
    missed_threshold : int
        Number of missed heartbeats before a node is marked unavailable.
    reaper_interval_s : float
        Seconds between reaper cycles.
    max_workers : int
        Maximum number of gRPC handler threads.

    Returns
    -------
    tuple[grpc.Server, NodeRegistry, threading.Event]
        The configured server (not yet started), the registry, and
        the reaper stop event.
    """
    registry = NodeRegistry(
        heartbeat_interval_s=heartbeat_interval_s,
        missed_threshold=missed_threshold,
    )
    heartbeat_interval_ms = int(heartbeat_interval_s * 1000)
    servicer = NodeServiceServicer(
        registry, heartbeat_interval_ms=heartbeat_interval_ms
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    stop_event = threading.Event()

    return server, registry, stop_event


def main():
    """
    Entry point for the orchestrator process.

    Parses command-line arguments, starts the gRPC server and reaper
    thread, and blocks until interrupted with SIGINT or SIGTERM.
    """
    parser = argparse.ArgumentParser(description="PiSSM Orchestrator")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help=f"gRPC server port (default: {DEFAULT_GRPC_PORT})",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL_S,
        help=f"Expected heartbeat interval in seconds (default: {DEFAULT_HEARTBEAT_INTERVAL_S})",
    )
    parser.add_argument(
        "--missed-threshold",
        type=int,
        default=DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
        help=f"Missed heartbeats before marking unavailable (default: {DEFAULT_MISSED_HEARTBEATS_THRESHOLD})",
    )
    parser.add_argument(
        "--reaper-interval",
        type=float,
        default=DEFAULT_REAPER_INTERVAL_S,
        help=f"Seconds between reaper cycles (default: {DEFAULT_REAPER_INTERVAL_S})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    server, registry, stop_event = create_server(
        port=args.port,
        heartbeat_interval_s=args.heartbeat_interval,
        missed_threshold=args.missed_threshold,
        reaper_interval_s=args.reaper_interval,
    )

    server.start()
    logger.info("Orchestrator started on port %d", args.port)
    logger.info(
        "Heartbeat interval=%.1fs, missed threshold=%d, timeout=%.1fs",
        args.heartbeat_interval,
        args.missed_threshold,
        registry.timeout_s,
    )

    reaper_thread = threading.Thread(
        target=run_reaper,
        args=(registry, args.reaper_interval, stop_event),
        daemon=True,
    )
    reaper_thread.start()

    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        stop_event.set()
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    shutdown_event.wait()
    server.stop(grace=5)
    logger.info("Orchestrator stopped")


if __name__ == "__main__":
    main()
