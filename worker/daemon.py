"""
Worker daemon entry point.

Starts the HeartbeatClient which periodically sends heartbeats to
the orchestrator, and a gRPC server hosting the InferenceService
for receiving model loading and inference requests. Blocks until
interrupted with SIGINT or SIGTERM.
"""

import argparse
import logging
import signal
import threading
from concurrent import futures

import grpc

from orchestrator.config import DEFAULT_HEARTBEAT_INTERVAL_S
from proto.generated import inference_pb2_grpc
from inference.service import InferenceServiceServicer
from worker.heartbeat import HeartbeatClient
from worker.system_info import get_node_id

logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_PORT = 50052


def main():
    """
    Entry point for the worker daemon process.

    Parses command-line arguments, starts the heartbeat client and
    inference gRPC server, and blocks until interrupted.
    """
    parser = argparse.ArgumentParser(description="PiSSM Worker Daemon")
    parser.add_argument(
        "--orchestrator",
        required=True,
        help="Orchestrator address (host:port)",
    )
    parser.add_argument(
        "--node-id",
        default=None,
        help="Override the node ID (default: hostname)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL_S,
        help=f"Initial heartbeat interval in seconds (default: {DEFAULT_HEARTBEAT_INTERVAL_S})",
    )
    parser.add_argument(
        "--inference-port",
        type=int,
        default=DEFAULT_INFERENCE_PORT,
        help=f"gRPC port for inference service (default: {DEFAULT_INFERENCE_PORT})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    node_id = args.node_id if args.node_id else get_node_id()

    client = HeartbeatClient(
        orchestrator_address=args.orchestrator,
        node_id=node_id,
        interval_s=args.heartbeat_interval,
    )

    inference_servicer = InferenceServiceServicer()
    inference_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        inference_servicer, inference_server
    )
    inference_server.add_insecure_port(f"[::]:{args.inference_port}")

    client.start()
    inference_server.start()

    logger.info(
        "Worker daemon started: node_id='%s', orchestrator=%s, inference_port=%d",
        node_id,
        args.orchestrator,
        args.inference_port,
    )

    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    shutdown_event.wait()
    client.stop()
    inference_server.stop(grace=5)
    logger.info("Worker daemon stopped")


if __name__ == "__main__":
    main()
