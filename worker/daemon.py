"""
Worker daemon entry point.

Starts the HeartbeatClient which periodically sends heartbeats to
the orchestrator, reporting this node's identity and hardware state.
Blocks until interrupted with SIGINT or SIGTERM.
"""

import argparse
import logging
import signal
import threading

from orchestrator.config import DEFAULT_HEARTBEAT_INTERVAL_S
from worker.heartbeat import HeartbeatClient
from worker.system_info import get_node_id

logger = logging.getLogger(__name__)


def main():
    """
    Entry point for the worker daemon process.

    Parses command-line arguments, starts the heartbeat client,
    and blocks until interrupted.
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

    client.start()
    logger.info(
        "Worker daemon started: node_id='%s', orchestrator=%s",
        node_id,
        args.orchestrator,
    )

    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    shutdown_event.wait()
    client.stop()
    logger.info("Worker daemon stopped")


if __name__ == "__main__":
    main()
