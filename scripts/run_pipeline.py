"""
Pipeline inference script for PiSSM distributed Mamba inference.

Starts a NodeService (for worker heartbeats) and a PipelineCallbackService
(for result delivery) on separate ports, waits for workers to register,
plans layer dispatch, distributes shard weights, and runs autoregressive
generation token by token across the worker pipeline.

Usage:
    python scripts/run_pipeline.py --prompt "The distributed cluster"
    python scripts/run_pipeline.py --manifest manifests/mamba-130m.yaml --prompt "Hello"
    python scripts/run_pipeline.py --port 50051 --callback-port 50060 --prompt "Hi"

Workers must be started pointing at --port (default 50051):
    pissm-worker --orchestrator <host>:50051 --inference-port 50052
"""

import argparse
import logging
import signal
import socket
import threading
import time
from concurrent import futures

import grpc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from inference.manifest import load_manifest
from orchestrator.config import (
    DEFAULT_GRPC_PORT,
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    DEFAULT_REAPER_INTERVAL_S,
)
from orchestrator.dispatch import plan_dispatch
from orchestrator.model_store import ModelStore
from orchestrator.node_registry import NodeRegistry
from orchestrator.pipeline import PipelineCallbackServicer, PipelineRunner, ResultStore
from orchestrator.service import NodeServiceServicer
from orchestrator.worker_client import _CHANNEL_OPTIONS
from proto.generated import inference_pb2_grpc, nodes_pb2_grpc

DEFAULT_MANIFEST = "manifests/mamba-130m.yaml"
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_WAIT_TIMEOUT_S = 60.0
DEFAULT_INFERENCE_TIMEOUT_S = 120.0
DEFAULT_CALLBACK_PORT = 50060

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def start_node_server(registry, port, heartbeat_interval_s, missed_threshold):
    """
    Start the NodeService gRPC server for worker heartbeats.

    Parameters
    ----------
    registry : NodeRegistry
        Registry to populate from heartbeats.
    port : int
        Port to bind.
    heartbeat_interval_s : float
        Expected heartbeat interval used for timeout calculation.
    missed_threshold : int
        Missed heartbeats before marking a node unavailable.

    Returns
    -------
    grpc.Server
        The running server.
    """
    servicer = NodeServiceServicer(
        registry,
        heartbeat_interval_ms=int(heartbeat_interval_s * 1000),
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


def start_callback_server(result_store, port):
    """
    Start the PipelineCallbackService gRPC server.

    Parameters
    ----------
    result_store : ResultStore
        Shared result store that PipelineRunner waits on.
    port : int
        Port to bind.

    Returns
    -------
    grpc.Server
        The running server.
    """
    servicer = PipelineCallbackServicer(result_store)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=_CHANNEL_OPTIONS,
    )
    inference_pb2_grpc.add_PipelineCallbackServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


def wait_for_workers(registry, timeout_s, expected=None):
    """
    Block until enough available workers are registered.

    If ``expected`` is given, waits until exactly that many nodes are
    available (or the timeout expires). If ``expected`` is None, returns
    as soon as at least one node is available.

    Parameters
    ----------
    registry : NodeRegistry
        The live registry.
    timeout_s : float
        Maximum seconds to wait.
    expected : int or None
        Number of nodes to wait for. None means any node will do.

    Returns
    -------
    list[NodeInfo]
        Available nodes.

    Raises
    ------
    RuntimeError
        If the required number of workers do not register before the timeout.
    """
    target = expected if expected is not None else 1
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        nodes = registry.list_nodes(status_filter="available")
        if len(nodes) >= target:
            return nodes
        time.sleep(1.0)
    nodes = registry.list_nodes(status_filter="available")
    if len(nodes) < target:
        raise RuntimeError(
            f"Only {len(nodes)}/{target} worker(s) registered after {timeout_s:.0f}s. "
            "Ensure workers are running and pointing at this server's --port."
        )
    return nodes


def main():
    """
    Run distributed pipeline inference.
    """
    parser = argparse.ArgumentParser(
        description="Distributed pipeline inference with PiSSM"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help=f"NodeService port for worker heartbeats (default: {DEFAULT_GRPC_PORT})",
    )
    parser.add_argument(
        "--callback-port",
        type=int,
        default=DEFAULT_CALLBACK_PORT,
        help=f"PipelineCallbackService port (default: {DEFAULT_CALLBACK_PORT})",
    )
    parser.add_argument(
        "--callback-host",
        default="",
        help="Hostname or IP for callback address sent to workers (default: system hostname)",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to model manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Input text prompt to generate from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Number of tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Number of worker nodes to wait for before starting (default: start as soon as any node is available)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT_S,
        help=f"Seconds to wait for workers to register (default: {DEFAULT_WAIT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=DEFAULT_INFERENCE_TIMEOUT_S,
        help=f"Per-call pipeline timeout in seconds (default: {DEFAULT_INFERENCE_TIMEOUT_S})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    callback_host = args.callback_host or _get_local_ip()
    callback_address = f"{callback_host}:{args.callback_port}"

    registry = NodeRegistry(
        heartbeat_interval_s=DEFAULT_HEARTBEAT_INTERVAL_S,
        missed_threshold=DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    )
    result_store = ResultStore()

    node_server = start_node_server(
        registry,
        args.port,
        DEFAULT_HEARTBEAT_INTERVAL_S,
        DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    )
    callback_server = start_callback_server(result_store, args.callback_port)

    stop_event = threading.Event()
    reaper_thread = threading.Thread(
        target=lambda: _reaper_loop(registry, stop_event),
        daemon=True,
    )
    reaper_thread.start()

    logger.info("NodeService on port %d", args.port)
    logger.info("PipelineCallbackService on %s", callback_address)

    def handle_signal(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    model_store = ModelStore()
    runner = None

    try:
        target_msg = f"{args.nodes} node(s)" if args.nodes else "any node"
        logger.info(
            "Waiting up to %.0fs for %s to register...", args.wait_timeout, target_msg
        )
        nodes = wait_for_workers(registry, args.wait_timeout, expected=args.nodes)
        logger.info("%d worker(s) available.", len(nodes))

        manifest = load_manifest(args.manifest)
        plan = plan_dispatch(manifest, registry)
        logger.info(
            "Dispatch plan: %d node(s), %d total layers",
            len(plan.assignments),
            plan.total_layers,
        )
        for a in plan.assignments:
            role = []
            if a.is_first:
                role.append("first")
            if a.is_last:
                role.append("last")
            logger.info(
                "  %s: layers [%d,%d) [%s]",
                a.node_id,
                a.layer_start,
                a.layer_end,
                "/".join(role) or "middle",
            )

        logger.info("Loading model '%s' on orchestrator...", manifest.name)
        model_store.load(manifest)
        logger.info("Model loaded. Distributing shards to workers...")

        runner = PipelineRunner(
            model_store=model_store,
            plan=plan,
            orchestrator_callback_address=callback_address,
            result_store=result_store,
            timeout_s=args.inference_timeout,
        )
        runner.load()
        logger.info("Shards distributed and loaded on all workers.")

        tokenizer = AutoTokenizer.from_pretrained(manifest.tokenizer)
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        logger.info(
            "Generating %d tokens (prompt: %d tokens)...",
            args.max_new_tokens,
            input_ids.shape[1],
        )

        with tqdm(total=args.max_new_tokens, unit="tok", dynamic_ncols=True) as pbar:
            for _ in range(args.max_new_tokens):
                result = runner.run_forward(input_ids)
                next_token = (
                    result.output_tensor[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)
                pbar.update(1)

        output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logger.info("Response: %s", output_text)

    finally:
        stop_event.set()
        if runner is not None:
            try:
                runner.unload()
            except Exception as e:
                logger.warning("Unload failed: %s", e)
        model_store.unload()
        node_server.stop(grace=2)
        callback_server.stop(grace=2)
        logger.info("Shutdown complete.")


def _reaper_loop(registry, stop_event):
    while not stop_event.is_set():
        registry.reap_stale_nodes()
        stop_event.wait(timeout=DEFAULT_REAPER_INTERVAL_S)


if __name__ == "__main__":
    main()
