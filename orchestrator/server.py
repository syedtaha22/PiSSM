"""
Orchestrator process entry point.

Starts the gRPC server with the NodeServiceServicer, launches a
background reaper thread that periodically marks unresponsive nodes
as unavailable, and serves the FastAPI HTTP API (consumed by the TUI
and WebUI) in the same process. The gRPC server runs on its own
thread pool; the HTTP API runs under uvicorn as the main blocking
call.
"""

import argparse
import logging
import threading
from concurrent import futures

import grpc
import uvicorn

from proto.generated import inference_pb2_grpc, nodes_pb2_grpc
from inference.model_registry import ModelRegistry
from orchestrator.config import (
    DEFAULT_GRPC_PORT,
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_HTTP_PORT,
    DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    DEFAULT_REAPER_INTERVAL_S,
)
from orchestrator.http_api import create_app
from orchestrator.node_registry import NodeRegistry
from orchestrator.pipeline import PipelineCallbackServicer, ResultStore
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
    Create and configure the orchestrator gRPC server and HTTP API.

    Registers both NodeService (heartbeat/registry) and
    PipelineCallbackService (result delivery) on the same gRPC server,
    and builds a FastAPI app wired to the same NodeRegistry and a new
    ModelRegistry.

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
    tuple[grpc.Server, NodeRegistry, ModelRegistry, threading.Event, ResultStore, FastAPI]
        The configured gRPC server (not yet started), the node registry,
        the model registry, the reaper stop event, the pipeline result
        store, and the FastAPI HTTP app (not yet running).
    """
    registry = NodeRegistry(
        heartbeat_interval_s=heartbeat_interval_s,
        missed_threshold=missed_threshold,
    )
    model_registry = ModelRegistry()
    heartbeat_interval_ms = int(heartbeat_interval_s * 1000)
    servicer = NodeServiceServicer(
        registry, heartbeat_interval_ms=heartbeat_interval_ms
    )

    result_store = ResultStore()
    callback_servicer = PipelineCallbackServicer(result_store)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    inference_pb2_grpc.add_PipelineCallbackServiceServicer_to_server(
        callback_servicer, server
    )
    server.add_insecure_port(f"[::]:{port}")

    stop_event = threading.Event()

    app = create_app(registry, model_registry)

    return server, registry, model_registry, stop_event, result_store, app


def main():
    """
    Entry point for the orchestrator process.

    Parses command-line arguments, starts the gRPC server and reaper
    thread in the background, then runs the FastAPI HTTP API under
    uvicorn as the main blocking call. uvicorn installs its own
    SIGINT/SIGTERM handlers and shuts down gracefully on either; once
    it returns, the gRPC server and reaper are stopped too.
    """
    parser = argparse.ArgumentParser(description="PiSSM Orchestrator")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help=f"gRPC server port (default: {DEFAULT_GRPC_PORT})",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"HTTP API port for the TUI/WebUI (default: {DEFAULT_HTTP_PORT})",
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

    server, registry, model_registry, stop_event, _, app = create_server(
        port=args.port,
        heartbeat_interval_s=args.heartbeat_interval,
        missed_threshold=args.missed_threshold,
        reaper_interval_s=args.reaper_interval,
    )

    server.start()
    logger.info("Orchestrator gRPC server started on port %d", args.port)
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

    logger.info("Orchestrator HTTP API starting on port %d", args.http_port)
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.http_port, log_level="info")
    finally:
        stop_event.set()
        server.stop(grace=5)
        logger.info("Orchestrator stopped")


if __name__ == "__main__":
    main()
