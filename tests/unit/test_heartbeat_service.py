"""
Tests for the NodeServiceServicer gRPC handler.

Validates that the servicer correctly bridges gRPC requests to the
NodeRegistry. Tests call servicer methods directly with mock context
objects, avoiding any network I/O.
"""

from unittest.mock import MagicMock

import grpc

from proto.generated import nodes_pb2
from orchestrator.node_registry import NodeRegistry
from orchestrator.service import NodeServiceServicer

STATUS_MAP = {
    "available": nodes_pb2.AVAILABLE,
    "busy": nodes_pb2.BUSY,
    "unavailable": nodes_pb2.UNAVAILABLE,
}


def make_heartbeat_request(
    node_id="node-1",
    ip_address="192.168.1.10",
    available_ram_mb=3800,
    total_ram_mb=4096,
    cpu_count=4,
    arch="aarch64",
    os_name="Linux",
    os_version="6.6.31+rpt-rpi-2712",
    timestamp=1000000,
):
    """
    Build a HeartbeatRequest with sensible defaults.

    Parameters
    ----------
    node_id : str
        Unique identifier for the node.
    ip_address : str
        IP address of the node.
    available_ram_mb : int
        Available RAM in megabytes.
    total_ram_mb : int
        Total RAM in megabytes.
    cpu_count : int
        Number of logical CPU cores.
    arch : str
        CPU architecture string.
    os_name : str
        Operating system name.
    os_version : str
        OS kernel version string.
    timestamp : int
        Unix timestamp in seconds.

    Returns
    -------
    nodes_pb2.HeartbeatRequest
        A populated heartbeat request.
    """
    return nodes_pb2.HeartbeatRequest(
        node_id=node_id,
        ip_address=ip_address,
        available_ram_mb=available_ram_mb,
        total_ram_mb=total_ram_mb,
        cpu_count=cpu_count,
        arch=arch,
        os_name=os_name,
        os_version=os_version,
        timestamp=timestamp,
    )


class TestHeartbeatRPC:
    """
    Tests for the Heartbeat RPC method.
    """

    def test_heartbeat_returns_acknowledged(self):
        """
        A valid heartbeat request returns acknowledged=True.
        """
        registry = NodeRegistry()
        servicer = NodeServiceServicer(registry)
        context = MagicMock()

        response = servicer.Heartbeat(make_heartbeat_request(), context)

        assert response.acknowledged is True

    def test_heartbeat_registers_node_in_registry(self):
        """
        After a heartbeat, the node exists in the registry with correct fields.
        """
        registry = NodeRegistry()
        servicer = NodeServiceServicer(registry)
        context = MagicMock()

        servicer.Heartbeat(make_heartbeat_request(), context)

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

    def test_heartbeat_returns_configured_interval(self):
        """
        The response carries the heartbeat interval configured on the servicer.
        """
        registry = NodeRegistry()
        servicer = NodeServiceServicer(registry, heartbeat_interval_ms=5000)
        context = MagicMock()

        response = servicer.Heartbeat(make_heartbeat_request(), context)

        assert response.heartbeat_interval_ms == 5000

    def test_heartbeat_default_interval(self):
        """
        The default heartbeat interval is 2000ms.
        """
        registry = NodeRegistry()
        servicer = NodeServiceServicer(registry)
        context = MagicMock()

        response = servicer.Heartbeat(make_heartbeat_request(), context)

        assert response.heartbeat_interval_ms == 2000


class TestReportStatusRPC:
    """
    Tests for the ReportStatus RPC method.
    """

    def test_report_status_returns_correct_fields(self):
        """
        ReportStatus for a known node returns all fields correctly.
        """
        registry = NodeRegistry()
        servicer = NodeServiceServicer(registry)
        context = MagicMock()

        servicer.Heartbeat(make_heartbeat_request(), context)

        request = nodes_pb2.StatusRequest(node_id="node-1")
        response = servicer.ReportStatus(request, context)

        assert response.node_id == "node-1"
        assert response.ip_address == "192.168.1.10"
        assert response.available_ram_mb == 3800
        assert response.total_ram_mb == 4096
        assert response.cpu_count == 4
        assert response.arch == "aarch64"
        assert response.os_name == "Linux"
        assert response.os_version == "6.6.31+rpt-rpi-2712"
        assert response.status == nodes_pb2.AVAILABLE

    def test_report_status_unknown_node_sets_not_found(self):
        """
        ReportStatus for an unknown node sets gRPC NOT_FOUND on the context.
        """
        registry = NodeRegistry()
        servicer = NodeServiceServicer(registry)
        context = MagicMock()

        request = nodes_pb2.StatusRequest(node_id="nonexistent")
        servicer.ReportStatus(request, context)

        context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        context.set_details.assert_called_once()
