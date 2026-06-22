"""
Integration tests for the gRPC heartbeat flow.

These tests spin up a real gRPC server in-process, send heartbeats
through a real gRPC channel, and verify the registry state. This
validates the full serialization/deserialization path that unit
tests with mock contexts do not cover.
"""

from proto.generated import nodes_pb2
from proto.generated import nodes_pb2_grpc


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


class TestHeartbeatRoundtrip:
    """
    Tests that heartbeats sent over a real gRPC channel reach the
    registry correctly.
    """

    def test_heartbeat_acknowledged(self, grpc_server_and_channel):
        """
        A heartbeat sent through the gRPC channel returns acknowledged=True.
        """
        _, channel, _ = grpc_server_and_channel
        stub = nodes_pb2_grpc.NodeServiceStub(channel)

        response = stub.Heartbeat(make_heartbeat_request())

        assert response.acknowledged is True
        assert response.heartbeat_interval_ms == 2000

    def test_multiple_heartbeats_same_node(self, grpc_server_and_channel):
        """
        Multiple heartbeats from the same node result in one registry entry.
        """
        _, channel, registry = grpc_server_and_channel
        stub = nodes_pb2_grpc.NodeServiceStub(channel)

        stub.Heartbeat(make_heartbeat_request(node_id="node-1"))
        stub.Heartbeat(make_heartbeat_request(node_id="node-1"))
        stub.Heartbeat(make_heartbeat_request(node_id="node-1"))

        nodes = registry.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "node-1"

    def test_two_different_nodes(self, grpc_server_and_channel):
        """
        Heartbeats from two different nodes create two registry entries.
        """
        _, channel, registry = grpc_server_and_channel
        stub = nodes_pb2_grpc.NodeServiceStub(channel)

        stub.Heartbeat(make_heartbeat_request(node_id="node-1"))
        stub.Heartbeat(
            make_heartbeat_request(node_id="node-2", ip_address="192.168.1.11")
        )

        nodes = registry.list_nodes()
        assert len(nodes) == 2
        ids = {n.node_id for n in nodes}
        assert ids == {"node-1", "node-2"}

    def test_heartbeat_data_reaches_registry(self, grpc_server_and_channel):
        """
        All fields from the heartbeat request are stored in the registry.
        """
        _, channel, registry = grpc_server_and_channel
        stub = nodes_pb2_grpc.NodeServiceStub(channel)

        stub.Heartbeat(make_heartbeat_request())

        node = registry.get_node("node-1")
        assert node is not None
        assert node.ip_address == "192.168.1.10"
        assert node.available_ram_mb == 3800
        assert node.total_ram_mb == 4096
        assert node.cpu_count == 4
        assert node.arch == "aarch64"
        assert node.os_name == "Linux"
        assert node.os_version == "6.6.31+rpt-rpi-2712"
        assert node.status == "available"


class TestReportStatusRoundtrip:
    """
    Tests that ReportStatus works over a real gRPC channel.
    """

    def test_report_status_known_node(self, grpc_server_and_channel):
        """
        ReportStatus returns correct fields for a node registered via heartbeat.
        """
        _, channel, _ = grpc_server_and_channel
        stub = nodes_pb2_grpc.NodeServiceStub(channel)

        stub.Heartbeat(make_heartbeat_request())
        response = stub.ReportStatus(nodes_pb2.StatusRequest(node_id="node-1"))

        assert response.node_id == "node-1"
        assert response.ip_address == "192.168.1.10"
        assert response.status == nodes_pb2.AVAILABLE

    def test_report_status_unknown_node_raises(self, grpc_server_and_channel):
        """
        ReportStatus for an unknown node raises a gRPC NOT_FOUND error.
        """
        import grpc as grpc_module

        _, channel, _ = grpc_server_and_channel
        stub = nodes_pb2_grpc.NodeServiceStub(channel)

        try:
            stub.ReportStatus(nodes_pb2.StatusRequest(node_id="nonexistent"))
            assert False, "Expected RpcError"
        except grpc_module.RpcError as e:
            assert e.code() == grpc_module.StatusCode.NOT_FOUND
