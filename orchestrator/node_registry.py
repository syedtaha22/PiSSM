"""
Thread-safe registry of worker nodes in the PiSSM cluster.

Tracks node identity, hardware state, and heartbeat timing. Supports
failure detection by reaping nodes that miss consecutive heartbeats,
and auto-rejoin when a previously unavailable node resumes heartbeats.
"""

import time
import threading
from dataclasses import dataclass

from orchestrator.config import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
)


@dataclass
class NodeInfo:
    """
    Snapshot of a worker node's current state.

    Parameters
    ----------
    node_id : str
        Unique identifier for the node, typically the hostname.
    ip_address : str
        The node's reachable IP address on the cluster network.
    available_ram_mb : int
        Currently available RAM in megabytes.
    total_ram_mb : int
        Total physical RAM in megabytes.
    cpu_count : int
        Number of logical CPU cores.
    arch : str
        CPU architecture string, e.g. "aarch64", "x86_64".
    os_name : str
        Operating system name, e.g. "Linux".
    os_version : str
        OS kernel version string.
    status : str
        One of "available", "busy", or "unavailable".
    last_heartbeat : float
        Monotonic clock value of the most recent heartbeat.
    first_seen : float
        Monotonic clock value when the node first registered.
    """

    node_id: str
    ip_address: str
    available_ram_mb: int
    total_ram_mb: int
    cpu_count: int
    arch: str
    os_name: str
    os_version: str
    status: str
    last_heartbeat: float
    first_seen: float


class NodeRegistry:
    """
    Thread-safe registry that tracks worker nodes via heartbeats.

    Parameters
    ----------
    heartbeat_interval_s : float
        Expected interval between heartbeats in seconds.
    missed_threshold : int
        Number of missed heartbeats before a node is marked unavailable.
    clock : callable
        A zero-argument callable returning a monotonic timestamp.
        Defaults to ``time.monotonic``. Inject a mock clock for testing.
    """

    def __init__(
        self,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        missed_threshold: int = DEFAULT_MISSED_HEARTBEATS_THRESHOLD,
        clock=time.monotonic,
    ) -> None:
        self._heartbeat_interval_s = heartbeat_interval_s
        self._missed_threshold = missed_threshold
        self._clock = clock
        self._nodes: dict[str, NodeInfo] = {}
        self._lock = threading.Lock()

    @property
    def timeout_s(self) -> float:
        """
        Seconds of silence before a node is marked unavailable.
        """
        return self._heartbeat_interval_s * self._missed_threshold

    def update_node(
        self,
        node_id: str,
        ip_address: str,
        available_ram_mb: int,
        total_ram_mb: int,
        cpu_count: int,
        arch: str,
        os_name: str,
        os_version: str,
    ) -> None:
        """
        Register a new node or update an existing node's heartbeat.

        If the node was previously marked unavailable, its status is
        restored to available (auto-rejoin, FR-NM-05).

        Parameters
        ----------
        node_id : str
            Unique identifier for the node.
        ip_address : str
            The node's reachable IP address.
        available_ram_mb : int
            Currently available RAM in megabytes.
        total_ram_mb : int
            Total physical RAM in megabytes.
        cpu_count : int
            Number of logical CPU cores.
        arch : str
            CPU architecture string.
        os_name : str
            Operating system name.
        os_version : str
            OS kernel version string.
        """
        now = self._clock()
        with self._lock:
            existing = self._nodes.get(node_id)
            first_seen = existing.first_seen if existing else now
            self._nodes[node_id] = NodeInfo(
                node_id=node_id,
                ip_address=ip_address,
                available_ram_mb=available_ram_mb,
                total_ram_mb=total_ram_mb,
                cpu_count=cpu_count,
                arch=arch,
                os_name=os_name,
                os_version=os_version,
                status="available",
                last_heartbeat=now,
                first_seen=first_seen,
            )

    def get_node(self, node_id: str) -> NodeInfo | None:
        """
        Return a snapshot of a node's info, or None if not found.

        Parameters
        ----------
        node_id : str
            The node to look up.
        """
        with self._lock:
            return self._nodes.get(node_id)

    def list_nodes(self, status_filter: str | None = None) -> list[NodeInfo]:
        """
        Return all registered nodes, optionally filtered by status.

        Parameters
        ----------
        status_filter : str or None
            If provided, only return nodes matching this status.
        """
        with self._lock:
            nodes = list(self._nodes.values())
        if status_filter is not None:
            nodes = [n for n in nodes if n.status == status_filter]
        return nodes

    def reap_stale_nodes(self) -> list[str]:
        """
        Mark nodes that have exceeded the heartbeat timeout as unavailable.

        Only nodes currently in "available" or "busy" status are checked.

        Returns
        -------
        list[str]
            Node IDs that were reaped.
        """
        now = self._clock()
        reaped = []
        with self._lock:
            for node in self._nodes.values():
                if node.status in ("available", "busy"):
                    if now - node.last_heartbeat > self.timeout_s:
                        node.status = "unavailable"
                        reaped.append(node.node_id)
        return reaped
