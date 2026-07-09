"""
Layer dispatch engine for pipeline-parallel inference.

Partitions a model's layer stack across available worker nodes and
produces shard assignments for the orchestrator to distribute via
LoadShard. Architecture-agnostic: callers pass a manifest with a
layer count and a live NodeRegistry; the engine returns a DispatchPlan
describing the full circular pipeline topology.
"""

from dataclasses import dataclass

from orchestrator.node_registry import NodeRegistry


class DispatchError(Exception):
    """
    Raised when the dispatch engine cannot form a valid shard assignment.

    Causes include: no available nodes in the registry, or more nodes
    than model layers.
    """


@dataclass(frozen=True)
class ShardAssignment:
    """
    Assignment of a contiguous layer range to a specific worker node.

    Parameters
    ----------
    node_id : str
        Registry identifier of the assigned worker.
    ip_address : str
        Reachable IP address of the worker on the cluster network.
    inference_port : int
        gRPC port on which the worker's InferenceService is listening.
    layer_start : int
        First layer index assigned to this shard (inclusive).
    layer_end : int
        Last layer index assigned to this shard (exclusive).
    is_first : bool
        True if this shard owns the embedding component and receives
        raw token IDs as input.
    is_last : bool
        True if this shard owns the final norm and lm_head and produces
        logits as output.
    next_worker_address : str
        Address ("host:port") of the next worker in the pipeline.
        Empty string for the last shard, which delivers results directly
        to the orchestrator's PipelineCallbackService.
    """

    node_id: str
    ip_address: str
    inference_port: int
    layer_start: int
    layer_end: int
    is_first: bool
    is_last: bool
    next_worker_address: str


@dataclass(frozen=True)
class DispatchPlan:
    """
    Complete shard topology for a single pipeline dispatch.

    Parameters
    ----------
    assignments : list[ShardAssignment]
        Ordered list of shard assignments from first to last in the
        pipeline. The orchestrator sends LoadShard in this order.
    arch : str
        Model architecture string from the manifest.
    model_name : str
        Model name from the manifest.
    total_layers : int
        Total number of layers in the full model.
    """

    assignments: list
    arch: str
    model_name: str
    total_layers: int


def split_layers(total_layers: int, num_nodes: int) -> list[tuple[int, int]]:
    """
    Partition [0, total_layers) into num_nodes contiguous non-overlapping ranges.

    The base size is total_layers // num_nodes. The last range absorbs any
    remainder so all layers are covered exactly once.

    Parameters
    ----------
    total_layers : int
        Total number of layers in the model.
    num_nodes : int
        Number of nodes to partition across.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples in ascending order, one per node.

    Raises
    ------
    DispatchError
        If num_nodes exceeds total_layers.
    """
    if num_nodes > total_layers:
        raise DispatchError(
            f"Cannot assign {num_nodes} nodes to {total_layers} layers: "
            "more nodes than layers."
        )
    base = total_layers // num_nodes
    ranges = []
    start = 0
    for i in range(num_nodes):
        end = total_layers if i == num_nodes - 1 else start + base
        ranges.append((start, end))
        start = end
    return ranges


def plan_dispatch(manifest, registry: NodeRegistry) -> DispatchPlan:
    """
    Build a DispatchPlan from a manifest and a live node registry.

    Queries all AVAILABLE nodes from the registry, partitions the
    model's layers with split_layers, and assigns one contiguous range
    to each node. Nodes are assigned in the order returned by
    registry.list_nodes (insertion order).

    Parameters
    ----------
    manifest : ModelManifest
        Manifest for the model to dispatch. Must expose ``layers``,
        ``arch``, and ``name`` attributes.
    registry : NodeRegistry
        Live registry of worker nodes.

    Returns
    -------
    DispatchPlan
        Ordered shard assignments covering the full model.

    Raises
    ------
    DispatchError
        If no AVAILABLE nodes are found, or if there are more nodes
        than model layers.
    """
    nodes = registry.list_nodes(status_filter="available")
    if not nodes:
        raise DispatchError("no available nodes in the registry")

    ranges = split_layers(manifest.layers, len(nodes))
    assignments = []
    for i, (node, (start, end)) in enumerate(zip(nodes, ranges)):
        is_last = i == len(nodes) - 1
        if is_last:
            next_addr = ""
        else:
            nxt = nodes[i + 1]
            next_addr = f"{nxt.ip_address}:{nxt.inference_port}"
        assignments.append(
            ShardAssignment(
                node_id=node.node_id,
                ip_address=node.ip_address,
                inference_port=node.inference_port,
                layer_start=start,
                layer_end=end,
                is_first=(i == 0),
                is_last=is_last,
                next_worker_address=next_addr,
            )
        )
    return DispatchPlan(
        assignments=assignments,
        arch=manifest.arch,
        model_name=manifest.name,
        total_layers=manifest.layers,
    )
