"""
Pipeline profiling script for PiSSM distributed Mamba inference.

Mirrors profile_single_node.py exactly: same token counts, same warmup
and run counts, same CSV columns plus num_nodes, node_latencies_ms, and
communication_overhead_ms. Forward pass and generation are profiled at
each token count. Generation runs the full circular pipeline once per new
token (naive; no cross-node SSM state reuse).

Results are appended to benchmarks/pipeline_baseline.csv.

Usage:
    python scripts/profile_pipeline.py
    python scripts/profile_pipeline.py --port 50051 --callback-port 50060
    python scripts/profile_pipeline.py --runs 20 --warmup 5
"""

import argparse
import csv
import json
import logging
import os
import signal
import socket
import statistics
import threading
import time
from concurrent import futures

import grpc
import torch
from tqdm import tqdm

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
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 3
DEFAULT_WAIT_TIMEOUT_S = 60.0
DEFAULT_INFERENCE_TIMEOUT_S = 120.0
DEFAULT_CALLBACK_PORT = 50060
TOKEN_COUNTS = [64, 128, 512, 1024]
MAX_NEW_TOKENS = 30
CSV_PATH = "benchmarks/pipeline_baseline.csv"

CSV_COLUMNS = [
    "timestamp",
    "hostname",
    "arch",
    "model_name",
    "tokens",
    "pass_type",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "p95_ms",
    "peak_memory_mb",
    "num_nodes",
    "node_latencies_ms",
    "communication_overhead_ms",
]

logger = logging.getLogger(__name__)


def compute_stats(latencies):
    """
    Compute summary statistics from a list of latency measurements.

    Parameters
    ----------
    latencies : list[float]
        Latency values in milliseconds.

    Returns
    -------
    dict
        Dictionary with mean, median, min, max, and p95 values.
    """
    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)
    return {
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "p95_ms": round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2),
    }


def profile_forward_pass(runner, input_ids, warmup, runs):
    """
    Profile pipeline forward pass latency.

    Parameters
    ----------
    runner : PipelineRunner
        The loaded pipeline runner.
    input_ids : torch.Tensor
        Input token IDs of shape (1, seq_len).
    warmup : int
        Number of warmup passes to discard.
    runs : int
        Number of timed passes to record.

    Returns
    -------
    tuple[list[float], list[list[float]], list[float], int]
        (wall_latencies_ms, per_run_node_latencies, comm_overheads_ms, peak_memory_mb)
    """
    for _ in tqdm(range(warmup), desc="  warmup", leave=False):
        runner.run_forward(input_ids)

    wall_latencies = []
    per_run_node_latencies = []
    comm_overheads = []
    peak_mem = 0

    with tqdm(total=runs, desc="  forward", unit="run", leave=False) as pbar:
        for _ in range(runs):
            start = time.perf_counter()
            result = runner.run_forward(input_ids)
            elapsed_ms = (time.perf_counter() - start) * 1000

            node_sum = sum(result.node_latencies_ms)
            wall_latencies.append(elapsed_ms)
            per_run_node_latencies.append(result.node_latencies_ms)
            comm_overheads.append(elapsed_ms - node_sum)
            if result.node_peak_memory_mb:
                peak_mem = max(peak_mem, max(result.node_peak_memory_mb))
            pbar.update(1)

    return wall_latencies, per_run_node_latencies, comm_overheads, peak_mem


def profile_generate(runner, input_ids, max_new_tokens, warmup, runs):
    """
    Profile autoregressive generation latency.

    Runs the full pipeline once per new token (naive - no cross-node
    SSM state reuse). Timing covers the entire generation loop.

    Parameters
    ----------
    runner : PipelineRunner
        The loaded pipeline runner.
    input_ids : torch.Tensor
        Input token IDs of shape (1, seq_len).
    max_new_tokens : int
        Tokens to generate per generation call.
    warmup : int
        Number of warmup passes to discard.
    runs : int
        Number of timed passes to record.

    Returns
    -------
    tuple[list[float], list[list[float]], list[float], int]
        (wall_latencies_ms, mean_node_latencies_per_run, comm_overheads_ms, peak_memory_mb)
    """

    def _generate(ids):
        cur = ids
        node_lats_accum = []
        peak = 0
        for _ in range(max_new_tokens):
            result = runner.run_forward(cur)
            next_tok = result.output_tensor[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            cur = torch.cat([cur, next_tok], dim=1)
            node_lats_accum.append(result.node_latencies_ms)
            if result.node_peak_memory_mb:
                peak = max(peak, max(result.node_peak_memory_mb))
        return node_lats_accum, peak

    for _ in tqdm(range(warmup), desc="  warmup", leave=False):
        _generate(input_ids)

    wall_latencies = []
    per_run_node_latencies = []
    comm_overheads = []
    peak_mem = 0

    with tqdm(
        total=runs,
        desc=f"  generate x{max_new_tokens}",
        unit="run",
        leave=False,
    ) as pbar:
        for _ in range(runs):
            start = time.perf_counter()
            node_lats_per_step, run_peak = _generate(input_ids)
            elapsed_ms = (time.perf_counter() - start) * 1000

            num_nodes = len(node_lats_per_step[0]) if node_lats_per_step else 0
            mean_node_lats = [
                statistics.mean(node_lats_per_step[s][n] for s in range(max_new_tokens))
                for n in range(num_nodes)
            ]
            node_sum = sum(mean_node_lats)
            wall_latencies.append(elapsed_ms)
            per_run_node_latencies.append(mean_node_lats)
            comm_overheads.append(elapsed_ms - node_sum * max_new_tokens)
            peak_mem = max(peak_mem, run_peak)
            pbar.update(1)

    return wall_latencies, per_run_node_latencies, comm_overheads, peak_mem


def start_servers(
    registry, node_port, callback_port, heartbeat_interval_s, missed_threshold
):
    """
    Start NodeService and PipelineCallbackService servers.

    Parameters
    ----------
    registry : NodeRegistry
        Node registry.
    node_port : int
        Port for NodeService.
    callback_port : int
        Port for PipelineCallbackService.
    heartbeat_interval_s : float
        Expected heartbeat interval.
    missed_threshold : int
        Missed heartbeats before marking unavailable.

    Returns
    -------
    tuple[grpc.Server, grpc.Server, ResultStore]
        (node_server, callback_server, result_store)
    """
    servicer = NodeServiceServicer(
        registry,
        heartbeat_interval_ms=int(heartbeat_interval_s * 1000),
    )
    node_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nodes_pb2_grpc.add_NodeServiceServicer_to_server(servicer, node_server)
    node_server.add_insecure_port(f"[::]:{node_port}")
    node_server.start()

    result_store = ResultStore()
    cb_servicer = PipelineCallbackServicer(result_store)
    callback_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=_CHANNEL_OPTIONS,
    )
    inference_pb2_grpc.add_PipelineCallbackServiceServicer_to_server(
        cb_servicer, callback_server
    )
    callback_server.add_insecure_port(f"[::]:{callback_port}")
    callback_server.start()

    return node_server, callback_server, result_store


def wait_for_workers(registry, timeout_s, expected=None):
    """
    Block until enough available workers are registered.

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
            f"Only {len(nodes)}/{target} worker(s) registered after {timeout_s:.0f}s."
        )
    return nodes


def main():
    """
    Run the pipeline profiling script.
    """
    parser = argparse.ArgumentParser(
        description="Profile PiSSM distributed pipeline inference"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help=f"NodeService port (default: {DEFAULT_GRPC_PORT})",
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
        help="Hostname/IP for callback address (default: system hostname)",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Model manifest path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Timed runs per configuration (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup passes per configuration (default: {DEFAULT_WARMUP})",
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
        help=f"Seconds to wait for workers (default: {DEFAULT_WAIT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=DEFAULT_INFERENCE_TIMEOUT_S,
        help=f"Per-call pipeline timeout in seconds (default: {DEFAULT_INFERENCE_TIMEOUT_S})",
    )
    parser.add_argument(
        "--output",
        default=CSV_PATH,
        help=f"CSV output path (default: {CSV_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    callback_host = args.callback_host or socket.gethostname()
    callback_address = f"{callback_host}:{args.callback_port}"

    registry = NodeRegistry(
        heartbeat_interval_s=DEFAULT_HEARTBEAT_INTERVAL_S,
        missed_threshold=DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    )

    stop_event = threading.Event()
    reaper_thread = threading.Thread(
        target=lambda: _reaper_loop(registry, stop_event),
        daemon=True,
    )
    reaper_thread.start()

    node_server, callback_server, result_store = start_servers(
        registry,
        args.port,
        args.callback_port,
        DEFAULT_HEARTBEAT_INTERVAL_S,
        DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
    )

    def handle_signal(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("NodeService on port %d", args.port)
    logger.info("PipelineCallbackService on %s", callback_address)

    model_store = ModelStore()
    runner = None

    try:
        target_msg = f"{args.nodes} node(s)" if args.nodes else "any node"
        logger.info(
            "Waiting up to %.0fs for %s to register...", args.wait_timeout, target_msg
        )
        nodes = wait_for_workers(registry, args.wait_timeout, expected=args.nodes)
        logger.info("%d worker(s) registered.", len(nodes))

        manifest = load_manifest(args.manifest)
        plan = plan_dispatch(manifest, registry)
        num_nodes = len(plan.assignments)
        logger.info(
            "Dispatch plan: %d node(s), %d layers", num_nodes, plan.total_layers
        )

        logger.info("Loading model '%s'...", manifest.name)
        model_store.load(manifest)
        logger.info(
            "Model loaded (~%d MB). Distributing shards...",
            model_store._handle.memory_mb,
        )

        runner = PipelineRunner(
            model_store=model_store,
            plan=plan,
            orchestrator_callback_address=callback_address,
            result_store=result_store,
            timeout_s=args.inference_timeout,
        )
        runner.load()
        logger.info("Shards loaded. Starting profiling...")

        results = []
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        hostname = socket.gethostname()

        for token_count in tqdm(TOKEN_COUNTS, desc="token counts", unit="cfg"):
            if stop_event.is_set():
                break

            input_ids = torch.randint(0, 50280, (1, token_count), dtype=torch.int64)

            fp_wall, fp_node_lats, fp_comm, fp_peak = profile_forward_pass(
                runner, input_ids, args.warmup, args.runs
            )
            fp_stats = compute_stats(fp_wall)
            mean_node_lats = (
                [
                    round(statistics.mean(run[i] for run in fp_node_lats), 2)
                    for i in range(num_nodes)
                ]
                if fp_node_lats and fp_node_lats[0]
                else []
            )
            mean_comm = round(statistics.mean(fp_comm), 2)
            results.append(
                {
                    "timestamp": timestamp,
                    "hostname": hostname,
                    "arch": manifest.arch,
                    "model_name": manifest.name,
                    "tokens": token_count,
                    "pass_type": "forward_pass",
                    **fp_stats,
                    "peak_memory_mb": fp_peak,
                    "num_nodes": num_nodes,
                    "node_latencies_ms": json.dumps(mean_node_lats),
                    "communication_overhead_ms": mean_comm,
                }
            )

            if stop_event.is_set():
                break

            gen_wall, gen_node_lats, gen_comm, gen_peak = profile_generate(
                runner, input_ids, MAX_NEW_TOKENS, args.warmup, args.runs
            )
            gen_stats = compute_stats(gen_wall)
            mean_gen_node_lats = (
                [
                    round(statistics.mean(run[i] for run in gen_node_lats), 2)
                    for i in range(num_nodes)
                ]
                if gen_node_lats and gen_node_lats[0]
                else []
            )
            mean_gen_comm = round(statistics.mean(gen_comm), 2)
            results.append(
                {
                    "timestamp": timestamp,
                    "hostname": hostname,
                    "arch": manifest.arch,
                    "model_name": manifest.name,
                    "tokens": token_count,
                    "pass_type": f"generate_{MAX_NEW_TOKENS}",
                    **gen_stats,
                    "peak_memory_mb": gen_peak,
                    "num_nodes": num_nodes,
                    "node_latencies_ms": json.dumps(mean_gen_node_lats),
                    "communication_overhead_ms": mean_gen_comm,
                }
            )

        if results:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            file_exists = os.path.exists(args.output)
            with open(args.output, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(results)
            logger.info("Results appended to %s", args.output)

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
