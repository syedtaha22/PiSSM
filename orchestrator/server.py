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
from pathlib import Path

import grpc
import uvicorn

from inference.manifest import ManifestError, load_manifest
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
from orchestrator.worker_client import _CHANNEL_OPTIONS
from proto.generated import inference_pb2_grpc, nodes_pb2_grpc
from worker.system_info import get_ip_address

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """
    Return this machine's LAN-facing IP address.

    Delegates to worker.system_info.get_ip_address(), which prefers a
    wired interface over Wi-Fi when both are present. This matters
    here because the resulting address is what workers use to deliver
    pipeline results back to this orchestrator - if it resolved to a
    Wi-Fi address on a machine where Ethernet is the intended cluster
    link, every result delivery would go over Wi-Fi regardless of how
    the workers themselves are reached.

    Returns
    -------
    str
        The node's IPv4 address as a dotted-quad string.
    """
    return get_ip_address()


def register_existing_manifests(
    model_registry: ModelRegistry, manifests_dir: Path
) -> None:
    """
    Auto-register every manifest found in a directory at startup.

    Lets a user drop a checkpoint's manifest into `manifests/` and have
    it show up already registered, without a manual submission step.
    Invalid manifests are logged and skipped rather than failing
    orchestrator startup.

    Parameters
    ----------
    model_registry : ModelRegistry
        The registry to populate.
    manifests_dir : Path
        Directory to scan for `*.yaml` manifest files.
    """
    if not manifests_dir.is_dir():
        return

    for path in sorted(manifests_dir.glob("*.yaml")):
        try:
            manifest = load_manifest(str(path))
            model_registry.register(manifest)
            logger.info("Auto-registered model '%s' from %s", manifest.name, path)
        except (ManifestError, ValueError) as err:
            logger.warning("Skipping manifest %s: %s", path, err)


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
    callback_host=None,
):
    """
    Create and configure the orchestrator gRPC server and HTTP API.

    Registers both NodeService (heartbeat/registry) and
    PipelineCallbackService (result delivery) on the same gRPC server,
    and builds a FastAPI app wired to the same NodeRegistry and a new
    ModelRegistry. The FastAPI app's POST /infer route sends this same
    gRPC server's address to workers as the pipeline callback address,
    since PipelineCallbackService is registered on it. Any manifest
    already present in the top-level `manifests/` directory is
    auto-registered into the ModelRegistry before the app is built.

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
    callback_host : str or None
        Hostname or IP workers should use to reach this server's
        PipelineCallbackService. Defaults to this machine's LAN-facing
        IP address, auto-detected via ``_get_local_ip()``.

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
    manifests_dir = Path(__file__).resolve().parent.parent / "manifests"
    register_existing_manifests(model_registry, manifests_dir)
    heartbeat_interval_ms = int(heartbeat_interval_s * 1000)
    servicer = NodeServiceServicer(
        registry, heartbeat_interval_ms=heartbeat_interval_ms
    )

    result_store = ResultStore()
    callback_servicer = PipelineCallbackServicer(result_store)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=_CHANNEL_OPTIONS
    )
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    inference_pb2_grpc.add_PipelineCallbackServiceServicer_to_server(
        callback_servicer, server
    )
    server.add_insecure_port(f"[::]:{port}")

    stop_event = threading.Event()

    callback_address = f"{callback_host or _get_local_ip()}:{port}"
    app = create_app(registry, model_registry, result_store, callback_address)

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
    parser.add_argument(
        "--callback-host",
        default="",
        help="Hostname or IP workers use to reach this server's pipeline "
        "callback (default: auto-detected LAN IP)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    server, registry, _model_registry, stop_event, _, app = create_server(
        port=args.port,
        heartbeat_interval_s=args.heartbeat_interval,
        missed_threshold=args.missed_threshold,
        reaper_interval_s=args.reaper_interval,
        callback_host=args.callback_host or None,
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
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.http_port,
            log_level="info",
            access_log=False,
        )
    finally:
        stop_event.set()
        server.stop(grace=5)
        logger.info("Orchestrator stopped")


if __name__ == "__main__":
    main()
