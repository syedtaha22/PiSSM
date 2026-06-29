"""
Single-node profiling script for Mamba-130M.

Measures forward pass and generation latency at 128 and 512 input
tokens, along with peak memory usage. Outputs results to both stdout
and a CSV file at benchmarks/single_node_baseline.csv. These numbers
are the single-node baseline against which multi-node pipeline
parallelism overhead is compared.

Usage:
    python scripts/profile_single_node.py
    python scripts/profile_single_node.py --manifest manifests/mamba-130m.yaml
    python scripts/profile_single_node.py --runs 20
"""

import argparse
import csv
import os
import socket
import statistics
import time

import psutil
import torch

from inference.loader import load_model, unload_model
from inference.manifest import load_manifest

DEFAULT_MANIFEST = "manifests/mamba-130m.yaml"
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 3
TOKEN_COUNTS = [64, 128, 512, 1024]
CSV_PATH = "benchmarks/single_node_baseline.csv"

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
]


def profile_forward_pass(model, input_ids, warmup, runs):
    """
    Profile raw forward pass latency.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    input_ids : torch.Tensor
        Input token IDs of shape (1, seq_len).
    warmup : int
        Number of warmup passes to discard.
    runs : int
        Number of timed passes to record.

    Returns
    -------
    tuple[list[float], int]
        A list of latency measurements in milliseconds, and peak
        memory in megabytes.
    """
    process = psutil.Process()

    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids)

    latencies = []
    peak_mem = 0
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(input_ids)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        mem = process.memory_info().rss // (1024 * 1024)
        peak_mem = max(peak_mem, mem)

    return latencies, peak_mem


def profile_generate(model, input_ids, max_new_tokens, warmup, runs):
    """
    Profile autoregressive generation latency.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    input_ids : torch.Tensor
        Input token IDs of shape (1, seq_len).
    max_new_tokens : int
        Number of new tokens to generate.
    warmup : int
        Number of warmup passes to discard.
    runs : int
        Number of timed passes to record.

    Returns
    -------
    tuple[list[float], int]
        A list of latency measurements in milliseconds, and peak
        memory in megabytes.
    """
    process = psutil.Process()

    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    latencies = []
    peak_mem = 0
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        mem = process.memory_info().rss // (1024 * 1024)
        peak_mem = max(peak_mem, mem)

    return latencies, peak_mem


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


def main():
    """
    Run the profiling script.
    """
    parser = argparse.ArgumentParser(
        description="Profile Mamba inference on a single node"
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to model manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of timed runs per configuration (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup passes (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--output",
        default=CSV_PATH,
        help=f"CSV output path (default: {CSV_PATH})",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"Loading model '{manifest.name}' from '{manifest.checkpoint}'...")
    handle = load_model(manifest)
    print(f"Model loaded: ~{handle.memory_mb} MB")

    results = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    hostname = socket.gethostname()

    for token_count in TOKEN_COUNTS:
        input_ids = torch.randint(
            0, handle.tokenizer.vocab_size, (1, token_count), dtype=torch.int64
        )

        print(f"\n--- {token_count} tokens ---")

        print(f"  Forward pass ({args.warmup} warmup, {args.runs} timed)...")
        fp_latencies, fp_peak = profile_forward_pass(
            handle.model, input_ids, args.warmup, args.runs
        )
        fp_stats = compute_stats(fp_latencies)
        print(
            f"  Forward pass: mean={fp_stats['mean_ms']:.1f}ms, "
            f"median={fp_stats['median_ms']:.1f}ms, "
            f"p95={fp_stats['p95_ms']:.1f}ms, "
            f"peak={fp_peak}MB"
        )

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
            }
        )

        print(f"  Generate 30 tokens ({args.warmup} warmup, {args.runs} timed)...")
        gen_latencies, gen_peak = profile_generate(
            handle.model, input_ids, 30, args.warmup, args.runs
        )
        gen_stats = compute_stats(gen_latencies)
        print(
            f"  Generate: mean={gen_stats['mean_ms']:.1f}ms, "
            f"median={gen_stats['median_ms']:.1f}ms, "
            f"p95={gen_stats['p95_ms']:.1f}ms, "
            f"peak={gen_peak}MB"
        )

        results.append(
            {
                "timestamp": timestamp,
                "hostname": hostname,
                "arch": manifest.arch,
                "model_name": manifest.name,
                "tokens": token_count,
                "pass_type": "generate_30",
                **gen_stats,
                "peak_memory_mb": gen_peak,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults appended to {args.output}")

    unload_model(handle)
    print("Model unloaded.")


if __name__ == "__main__":
    main()
