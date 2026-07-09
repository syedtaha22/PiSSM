"""
Tests for the layer dispatch engine.

Covers split_layers (contiguous layer range partitioning) and
plan_dispatch (full shard assignment from a live NodeRegistry).
Both functions are pure logic with no network or model I/O.
"""

import pytest
from unittest.mock import MagicMock

from orchestrator.dispatch import (
    DispatchError,
    plan_dispatch,
    split_layers,
)
from orchestrator.node_registry import NodeRegistry


def make_registry(*nodes):
    """
    Build a NodeRegistry pre-populated with the given nodes.

    All nodes are inserted with minimal hardware metadata and end up
    in AVAILABLE status.

    Parameters
    ----------
    *nodes : tuple[str, str, int]
        Each tuple is (node_id, ip_address, inference_port).

    Returns
    -------
    NodeRegistry
        Registry with all provided nodes in AVAILABLE status.
    """
    registry = NodeRegistry()
    for node_id, ip, port in nodes:
        registry.update_node(
            node_id=node_id,
            ip_address=ip,
            available_ram_mb=3800,
            total_ram_mb=4096,
            cpu_count=4,
            arch="aarch64",
            os_name="Linux",
            os_version="6.6.31",
            inference_port=port,
        )
    return registry


def make_manifest(layers=24, arch="mamba", name="mamba-130m"):
    """
    Build a mock manifest with the given attributes.

    Parameters
    ----------
    layers : int
        Total number of model layers.
    arch : str
        Model architecture string.
    name : str
        Model name.

    Returns
    -------
    MagicMock
        Mock with layers, arch, and name attributes set.
    """
    manifest = MagicMock()
    manifest.layers = layers
    manifest.arch = arch
    manifest.name = name
    return manifest


# ---------------------------------------------------------------------------
# split_layers
# ---------------------------------------------------------------------------


class TestSplitLayers:
    """
    Tests for the split_layers partitioning function.
    """

    def test_even_split(self):
        """
        Even total_layers / num_nodes produces equal-sized ranges.
        """
        result = split_layers(total_layers=24, num_nodes=2)
        assert result == [(0, 12), (12, 24)]

    def test_uneven_split_last_takes_remainder(self):
        """
        When layers do not divide evenly, the last range absorbs the remainder.
        """
        result = split_layers(total_layers=10, num_nodes=3)
        assert result == [(0, 3), (3, 6), (6, 10)]

    def test_single_node_gets_full_range(self):
        """
        A single node receives the full [0, total_layers) range.
        """
        result = split_layers(total_layers=24, num_nodes=1)
        assert result == [(0, 24)]

    def test_raises_when_more_nodes_than_layers(self):
        """
        Raises DispatchError when num_nodes exceeds total_layers.
        """
        with pytest.raises(DispatchError, match="nodes"):
            split_layers(total_layers=2, num_nodes=3)


# ---------------------------------------------------------------------------
# plan_dispatch
# ---------------------------------------------------------------------------


class TestPlanDispatch:
    """
    Tests for the plan_dispatch shard assignment function.

    Nodes are assigned in the order returned by
    registry.list_nodes(status_filter="available"), which follows
    insertion order.
    """

    def test_correct_layer_ranges(self):
        """
        Each assignment receives the correct contiguous layer range.
        """
        registry = make_registry(
            ("node-0", "192.168.1.10", 50052),
            ("node-1", "192.168.1.11", 50052),
        )
        plan = plan_dispatch(make_manifest(layers=24), registry)

        ranges = [(a.layer_start, a.layer_end) for a in plan.assignments]
        assert ranges == [(0, 12), (12, 24)]

    def test_is_first_and_is_last_flags(self):
        """
        First assignment has is_first=True; last has is_last=True.
        Middle assignments have both False.
        """
        registry = make_registry(
            ("node-0", "192.168.1.10", 50052),
            ("node-1", "192.168.1.11", 50052),
        )
        plan = plan_dispatch(make_manifest(layers=24), registry)

        assert plan.assignments[0].is_first is True
        assert plan.assignments[0].is_last is False
        assert plan.assignments[-1].is_first is False
        assert plan.assignments[-1].is_last is True

    def test_next_worker_address_points_to_next_node(self):
        """
        next_worker_address on a non-last assignment is "ip:port" of the
        immediately following node.
        """
        registry = make_registry(
            ("node-0", "192.168.1.10", 50052),
            ("node-1", "192.168.1.11", 50053),
        )
        plan = plan_dispatch(make_manifest(layers=24), registry)

        assert plan.assignments[0].next_worker_address == "192.168.1.11:50053"

    def test_last_assignment_has_empty_next_worker_address(self):
        """
        The last assignment's next_worker_address is an empty string.
        """
        registry = make_registry(
            ("node-0", "192.168.1.10", 50052),
            ("node-1", "192.168.1.11", 50053),
        )
        plan = plan_dispatch(make_manifest(layers=24), registry)

        assert plan.assignments[-1].next_worker_address == ""

    def test_raises_with_empty_registry(self):
        """
        Raises DispatchError when the registry has no available nodes.
        """
        with pytest.raises(DispatchError, match="no available"):
            plan_dispatch(make_manifest(layers=24), NodeRegistry())

    def test_uses_all_available_nodes(self):
        """
        plan_dispatch produces exactly one assignment per available node.
        """
        registry = make_registry(
            ("node-0", "192.168.1.10", 50052),
            ("node-1", "192.168.1.11", 50052),
            ("node-2", "192.168.1.12", 50052),
        )
        plan = plan_dispatch(make_manifest(layers=24), registry)

        assert len(plan.assignments) == 3

    def test_single_node_is_both_first_and_last(self):
        """
        With a single node, is_first and is_last are both True and
        next_worker_address is empty.
        """
        registry = make_registry(("node-0", "192.168.1.10", 50052))
        plan = plan_dispatch(make_manifest(layers=24), registry)

        assert len(plan.assignments) == 1
        assert plan.assignments[0].is_first is True
        assert plan.assignments[0].is_last is True
        assert plan.assignments[0].next_worker_address == ""

    def test_middle_node_has_no_first_or_last_flag(self):
        """
        A middle assignment has is_first=False and is_last=False.
        """
        registry = make_registry(
            ("node-0", "192.168.1.10", 50052),
            ("node-1", "192.168.1.11", 50052),
            ("node-2", "192.168.1.12", 50052),
        )
        plan = plan_dispatch(make_manifest(layers=24), registry)

        assert plan.assignments[1].is_first is False
        assert plan.assignments[1].is_last is False

    def test_plan_carries_manifest_metadata(self):
        """
        DispatchPlan stores arch, model_name, and total_layers from the manifest.
        """
        registry = make_registry(("node-0", "192.168.1.10", 50052))
        plan = plan_dispatch(
            make_manifest(layers=24, arch="mamba", name="mamba-130m"), registry
        )

        assert plan.arch == "mamba"
        assert plan.model_name == "mamba-130m"
        assert plan.total_layers == 24
