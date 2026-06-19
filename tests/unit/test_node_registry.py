"""
Tests for the NodeRegistry and NodeInfo classes.

Covers FR-NM-02 (node registry), FR-NM-03 (failure detection),
FR-NM-04 (node listing), and FR-NM-05 (auto-rejoin).
"""

import threading

from orchestrator.node_registry import NodeRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_clock(start=0.0):
    """
    Return a controllable clock function.

    The returned clock starts at ``start`` and advances only when
    ``clock.advance(seconds)`` is called. This avoids any real
    ``time.sleep`` in unit tests.

    Parameters
    ----------
    start : float
        Initial time value for the clock.

    Returns
    -------
    callable
        A clock function with an ``advance(seconds)`` method.
    """
    state = {"now": start}

    def clock():
        return state["now"]

    def advance(seconds):
        state["now"] += seconds

    clock.advance = advance
    return clock


def register_node(
    registry,
    node_id="node-1",
    ip="192.168.1.10",
    available_ram=3800,
    total_ram=4096,
    cpu_count=4,
    arch="aarch64",
    os_name="Linux",
    os_version="6.6.31+rpt-rpi-2712",
):
    """
    Register a node with sensible defaults.

    Parameters
    ----------
    registry : NodeRegistry
        The registry to update.
    node_id : str
        Unique identifier for the node.
    ip : str
        IP address of the node.
    available_ram : int
        Available RAM in megabytes.
    total_ram : int
        Total RAM in megabytes.
    cpu_count : int
        Number of logical CPU cores.
    arch : str
        CPU architecture string.
    os_name : str
        Operating system name.
    os_version : str
        OS kernel version string.
    """
    registry.update_node(
        node_id=node_id,
        ip_address=ip,
        available_ram_mb=available_ram,
        total_ram_mb=total_ram,
        cpu_count=cpu_count,
        arch=arch,
        os_name=os_name,
        os_version=os_version,
    )


# ---------------------------------------------------------------------------
# FR-NM-02: Node registry  - registration, update, lookup
# ---------------------------------------------------------------------------


class TestNodeRegistration:
    """
    Tests for registering and looking up nodes.
    """

    def test_register_new_node(self):
        """
        Registering a new node stores all fields and sets status to available.
        """
        clock = make_clock(start=100.0)
        registry = NodeRegistry(clock=clock)

        register_node(registry)
        node = registry.get_node("node-1")

        assert node is not None
        assert node.node_id == "node-1"
        assert node.ip_address == "192.168.1.10"
        assert node.available_ram_mb == 3800
        assert node.total_ram_mb == 4096
        assert node.cpu_count == 4
        assert node.arch == "aarch64"
        assert node.os_name == "Linux"
        assert node.os_version == "6.6.31+rpt-rpi-2712"
        assert node.status == "available"
        assert node.last_heartbeat == 100.0
        assert node.first_seen == 100.0

    def test_update_existing_node(self):
        """
        Updating an existing node refreshes RAM and timestamp but preserves first_seen.
        """
        clock = make_clock(start=100.0)
        registry = NodeRegistry(clock=clock)

        register_node(registry, available_ram=3800)
        clock.advance(2.0)
        register_node(registry, available_ram=3500)

        node = registry.get_node("node-1")
        assert node.available_ram_mb == 3500
        assert node.last_heartbeat == 102.0
        assert node.first_seen == 100.0

    def test_get_node_unknown(self):
        """
        Looking up a node that was never registered returns None.
        """
        registry = NodeRegistry()
        assert registry.get_node("nonexistent") is None


# ---------------------------------------------------------------------------
# FR-NM-04: Node listing
# ---------------------------------------------------------------------------


class TestNodeListing:
    """
    Tests for listing nodes with optional status filtering.
    """

    def test_list_nodes_empty(self):
        """
        An empty registry returns an empty list.
        """
        registry = NodeRegistry()
        assert registry.list_nodes() == []

    def test_list_nodes_returns_all(self):
        """
        All registered nodes are returned when no filter is applied.
        """
        registry = NodeRegistry()
        register_node(registry, node_id="node-1")
        register_node(registry, node_id="node-2")
        register_node(registry, node_id="node-3")

        nodes = registry.list_nodes()
        assert len(nodes) == 3
        ids = {n.node_id for n in nodes}
        assert ids == {"node-1", "node-2", "node-3"}

    def test_list_nodes_filter_by_status(self):
        """
        Filtering by status returns only nodes matching that status.
        """
        clock = make_clock(start=0.0)
        registry = NodeRegistry(
            heartbeat_interval_s=1.0,
            missed_threshold=3,
            clock=clock,
        )

        register_node(registry, node_id="node-1")
        register_node(registry, node_id="node-2")

        clock.advance(4.0)
        register_node(registry, node_id="node-1")
        registry.reap_stale_nodes()

        available = registry.list_nodes(status_filter="available")
        unavailable = registry.list_nodes(status_filter="unavailable")

        assert len(available) == 1
        assert available[0].node_id == "node-1"
        assert len(unavailable) == 1
        assert unavailable[0].node_id == "node-2"


# ---------------------------------------------------------------------------
# FR-NM-03: Failure detection  - reaping stale nodes
# ---------------------------------------------------------------------------


class TestFailureDetection:
    """
    Tests for marking unresponsive nodes as unavailable.
    """

    def test_reap_stale_node(self):
        """
        A node that exceeds the timeout is marked unavailable.
        """
        clock = make_clock(start=0.0)
        registry = NodeRegistry(
            heartbeat_interval_s=2.0,
            missed_threshold=3,
            clock=clock,
        )

        register_node(registry)
        clock.advance(7.0)
        reaped = registry.reap_stale_nodes()

        assert reaped == ["node-1"]
        assert registry.get_node("node-1").status == "unavailable"

    def test_reap_does_not_affect_fresh_nodes(self):
        """
        Nodes within the timeout window are not reaped.
        """
        clock = make_clock(start=0.0)
        registry = NodeRegistry(
            heartbeat_interval_s=2.0,
            missed_threshold=3,
            clock=clock,
        )

        register_node(registry)
        clock.advance(1.0)
        reaped = registry.reap_stale_nodes()

        assert reaped == []
        assert registry.get_node("node-1").status == "available"

    def test_reap_does_not_reap_already_unavailable(self):
        """
        Already unavailable nodes are not reported again on subsequent reaps.
        """
        clock = make_clock(start=0.0)
        registry = NodeRegistry(
            heartbeat_interval_s=2.0,
            missed_threshold=3,
            clock=clock,
        )

        register_node(registry)
        clock.advance(7.0)
        first_reap = registry.reap_stale_nodes()
        second_reap = registry.reap_stale_nodes()

        assert first_reap == ["node-1"]
        assert second_reap == []

    def test_timeout_calculation(self):
        """
        The timeout is heartbeat_interval_s multiplied by missed_threshold.
        """
        registry = NodeRegistry(heartbeat_interval_s=2.0, missed_threshold=3)
        assert registry.timeout_s == 6.0


# ---------------------------------------------------------------------------
# FR-NM-05: Auto-rejoin
# ---------------------------------------------------------------------------


class TestAutoRejoin:
    """
    Tests for nodes automatically rejoining after being marked unavailable.
    """

    def test_rejoin_after_unavailable(self):
        """
        A node marked unavailable is restored to available on the next heartbeat.
        """
        clock = make_clock(start=0.0)
        registry = NodeRegistry(
            heartbeat_interval_s=2.0,
            missed_threshold=3,
            clock=clock,
        )

        register_node(registry)
        clock.advance(7.0)
        registry.reap_stale_nodes()
        assert registry.get_node("node-1").status == "unavailable"

        clock.advance(1.0)
        register_node(registry)
        node = registry.get_node("node-1")

        assert node.status == "available"
        assert node.last_heartbeat == 8.0
        assert node.first_seen == 0.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """
    Verify concurrent access does not corrupt registry state.
    """

    def test_concurrent_updates(self):
        """
        Multiple threads updating different nodes simultaneously should not
        raise exceptions or lose entries.
        """
        registry = NodeRegistry()
        errors = []

        def update_loop(node_id, count):
            try:
                for _ in range(count):
                    register_node(registry, node_id=node_id)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=update_loop, args=(f"node-{i}", 100))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(registry.list_nodes()) == 10
