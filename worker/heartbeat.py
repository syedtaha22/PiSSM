"""
Heartbeat client for worker nodes.

Periodically sends gRPC heartbeats to the orchestrator, reporting
the node's identity and hardware state. Handles server unavailability
gracefully by logging warnings and retrying on the next interval.
"""

import logging
import time
import threading

import grpc

from proto.generated import nodes_pb2
from proto.generated import nodes_pb2_grpc
from worker.system_info import (
    get_available_ram_mb,
    get_total_ram_mb,
    get_cpu_count,
    get_arch,
    get_ip_address,
    get_os_name,
    get_os_version,
)

logger = logging.getLogger(__name__)


class HeartbeatClient:
    """
    Periodically sends heartbeats to the orchestrator over gRPC.

    Parameters
    ----------
    orchestrator_address : str
        Address of the orchestrator gRPC server (host:port).
    node_id : str
        This node's unique identifier.
    interval_s : float
        Seconds between heartbeats.
    """

    def __init__(
        self,
        orchestrator_address: str,
        node_id: str,
        interval_s: float = 2.0,
    ) -> None:
        self._orchestrator_address = orchestrator_address
        self._node_id = node_id
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = None
        self._channel = None

    @property
    def is_running(self) -> bool:
        """
        Whether the heartbeat loop is currently active.
        """
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """
        Start the heartbeat loop in a background thread.
        """
        self._stop_event.clear()
        self._channel = grpc.insecure_channel(self._orchestrator_address)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Heartbeat client started for node '%s' -> %s (interval=%.1fs)",
            self._node_id,
            self._orchestrator_address,
            self._interval_s,
        )

    def stop(self) -> None:
        """
        Stop the heartbeat loop and close the gRPC channel.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 1.0)
            self._thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        logger.info("Heartbeat client stopped for node '%s'", self._node_id)

    def _heartbeat_loop(self) -> None:
        """
        Internal loop that sends heartbeats at the configured interval.

        Handles gRPC errors gracefully by logging and retrying on the
        next interval. Never crashes. Respects interval updates from
        the orchestrator's response.
        """
        stub = nodes_pb2_grpc.NodeServiceStub(self._channel)

        while not self._stop_event.is_set():
            try:
                request = nodes_pb2.HeartbeatRequest(
                    node_id=self._node_id,
                    ip_address=get_ip_address(),
                    available_ram_mb=get_available_ram_mb(),
                    total_ram_mb=get_total_ram_mb(),
                    cpu_count=get_cpu_count(),
                    arch=get_arch(),
                    os_name=get_os_name(),
                    os_version=get_os_version(),
                    timestamp=int(time.time()),
                )
                response = stub.Heartbeat(request, timeout=self._interval_s)

                if response.heartbeat_interval_ms > 0:
                    new_interval = response.heartbeat_interval_ms / 1000.0
                    if new_interval != self._interval_s:
                        logger.info(
                            "Heartbeat interval adjusted to %.1fs",
                            new_interval,
                        )
                        self._interval_s = new_interval

                logger.debug(
                    "Heartbeat sent for node '%s', acknowledged=%s",
                    self._node_id,
                    response.acknowledged,
                )

            except grpc.RpcError as e:
                logger.warning(
                    "Heartbeat failed for node '%s': %s",
                    self._node_id,
                    e,
                )

            self._stop_event.wait(timeout=self._interval_s)
