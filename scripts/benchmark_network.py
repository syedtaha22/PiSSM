"""
Network latency and throughput benchmark for PiSSM worker nodes.

Sends Ping RPCs at several payload sizes to a running worker's
InferenceService and records round-trip latency and throughput. Payload
sizes span small control messages (0 B) up to large activation payloads
(64 MB). Results are appended to benchmarks/network_baseline.csv.

The worker's InferenceService must be running before this script is
started. Ping requires no model to be loaded; it echoes the payload
back immediately.

Usage:
    python scripts/benchmark_network.py --worker 192.168.1.10:50052
    python scripts/benchmark_network.py --worker localhost:50052 --runs 20
"""

import argparse
import csv
import logging
import os
import socket
import statistics
import time

import grpc
from tqdm import tqdm

from orchestrator.worker_client import _CHANNEL_OPTIONS
from proto.generated import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

DEFAULT_RUNS = 10
DEFAULT_WARMUP = 3
CSV_PATH = "benchmarks/network_baseline.csv"

PAYLOAD_SIZES = [
    0,
    1 * 1024,
    64 * 1024,
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
]

CSV_COLUMNS = [
    "timestamp",
    "hostname",
    "worker_address",
    "payload_bytes",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "p95_ms",
    "throughput_mbps",
]


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


def ping_worker(stub, payload_size_bytes, warmup, runs):
    """
    Send Ping RPCs and record round-trip latency.

    Parameters
    ----------
    stub : inference_pb2_grpc.InferenceServiceStub
        gRPC stub connected to the worker.
    payload_size_bytes : int
        Number of bytes in the ping payload.
    warmup : int
        Number of warmup pings to discard.
    runs : int
        Number of timed pings to record.

    Returns
    -------
    list[float]
        Round-trip latency in milliseconds for each timed ping.
    """
    payload = b"\x00" * payload_size_bytes
    request = inference_pb2.PingRequest(payload=payload)

    for _ in range(warmup):
        stub.Ping(request)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        stub.Ping(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    return latencies


def format_bytes(n):
    """
    Format a byte count as a human-readable string.

    Parameters
    ----------
    n : int
        Number of bytes.

    Returns
    -------
    str
        Formatted string such as "64 KB" or "16 MB".
    """
    if n == 0:
        return "0 B"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n // (1024 * 1024)} MB"


def main():
    """
    Run the network benchmark script.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark network latency and throughput to a PiSSM worker"
    )
    parser.add_argument(
        "--worker",
        required=True,
        help="Worker InferenceService address (host:port)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Timed pings per payload size (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup pings per payload size (default: {DEFAULT_WARMUP})",
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

    logger.info("Connecting to worker at %s...", args.worker)
    channel = grpc.insecure_channel(args.worker, options=_CHANNEL_OPTIONS)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    results = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    hostname = socket.gethostname()

    for payload_size in tqdm(PAYLOAD_SIZES, desc="payload sizes", unit="size"):
        latencies = ping_worker(stub, payload_size, args.warmup, args.runs)
        stats = compute_stats(latencies)

        throughput_mbps = 0.0
        if payload_size > 0 and stats["mean_ms"] > 0:
            bytes_per_rtt = payload_size * 2
            throughput_mbps = round(
                bytes_per_rtt / (stats["mean_ms"] / 1000) / (1024 * 1024), 2
            )

        results.append(
            {
                "timestamp": timestamp,
                "hostname": hostname,
                "worker_address": args.worker,
                "payload_bytes": payload_size,
                **stats,
                "throughput_mbps": throughput_mbps,
            }
        )

    channel.close()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    logger.info("Results appended to %s", args.output)


if __name__ == "__main__":
    main()
