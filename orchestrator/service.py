"""
gRPC servicer implementation for the NodeService.

Bridges incoming gRPC heartbeat and status requests to the
NodeRegistry. This is the server-side handler that workers
communicate with over the network.
"""

import logging

import grpc

from proto.generated import nodes_pb2
from proto.generated import nodes_pb2_grpc
from orchestrator.node_registry import NodeRegistry

logger = logging.getLogger(__name__)

STATUS_TO_PROTO = {
    "unknown": nodes_pb2.UNKNOWN,
    "available": nodes_pb2.AVAILABLE,
    "busy": nodes_pb2.BUSY,
    "unavailable": nodes_pb2.UNAVAILABLE,
}


class NodeServiceServicer(nodes_pb2_grpc.NodeServiceServicer):
    """
    Handles Heartbeat and ReportStatus RPCs for the orchestrator.

    Parameters
    ----------
    registry : NodeRegistry
        The node registry to update on heartbeats and query on status requests.
    heartbeat_interval_ms : int
        The heartbeat interval to communicate back to workers in responses.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        heartbeat_interval_ms: int = 2000,
    ) -> None:
        self._registry = registry
        self._heartbeat_interval_ms = heartbeat_interval_ms

    def Heartbeat(self, request, context):
        """
        Process a heartbeat from a worker node.

        Registers the node if new, or updates its state if already known.

        Parameters
        ----------
        request : nodes_pb2.HeartbeatRequest
            The heartbeat payload from the worker.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        nodes_pb2.HeartbeatResponse
            Acknowledgement and the configured heartbeat interval.
        """
        self._registry.update_node(
            node_id=request.node_id,
            ip_address=request.ip_address,
            available_ram_mb=request.available_ram_mb,
            total_ram_mb=request.total_ram_mb,
            cpu_count=request.cpu_count,
            arch=request.arch,
            os_name=request.os_name,
            os_version=request.os_version,
            inference_port=request.inference_port,
        )

        logger.info(
            "Heartbeat from %s (%s), RAM: %d/%d MB",
            request.node_id,
            request.ip_address,
            request.available_ram_mb,
            request.total_ram_mb,
        )

        return nodes_pb2.HeartbeatResponse(
            acknowledged=True,
            heartbeat_interval_ms=self._heartbeat_interval_ms,
        )

    def ReportStatus(self, request, context):
        """
        Return the current status of a specific node.

        Parameters
        ----------
        request : nodes_pb2.StatusRequest
            Contains the node_id to query.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        nodes_pb2.StatusResponse
            The node's current state, or an empty response with
            NOT_FOUND if the node is unknown.
        """
        node = self._registry.get_node(request.node_id)

        if node is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Node '{request.node_id}' not found")
            return nodes_pb2.StatusResponse()

        return nodes_pb2.StatusResponse(
            node_id=node.node_id,
            ip_address=node.ip_address,
            available_ram_mb=node.available_ram_mb,
            total_ram_mb=node.total_ram_mb,
            cpu_count=node.cpu_count,
            arch=node.arch,
            os_name=node.os_name,
            os_version=node.os_version,
            status=STATUS_TO_PROTO.get(node.status, nodes_pb2.UNKNOWN),
            loaded_shards=0,
        )
