"""
Pipeline coordinator for circular pipeline-parallel inference.

ResultStore holds per-request futures that the last worker resolves via
DeliverResult. PipelineCallbackServicer is the gRPC handler that bridges
the incoming DeliverResult RPC to the ResultStore. PipelineRunner
orchestrates LoadShard, the initial RunShard fire-and-forward, and
UnloadShard across all workers in a DispatchPlan.
"""

import threading
import uuid
from dataclasses import dataclass

import torch

from inference.tensor_utils import deserialize_tensor, serialize_tensor
from orchestrator.worker_client import WorkerClient
from proto.generated import inference_pb2, inference_pb2_grpc


@dataclass
class PipelineResult:
    """
    Result of a single pipeline forward pass.

    Parameters
    ----------
    output_tensor : torch.Tensor
        The final output tensor produced by the last shard.
    node_latencies_ms : list[float]
        Per-node compute times in milliseconds, one entry per worker in
        pipeline order.
    node_peak_memory_mb : list[int]
        Per-node peak RSS memory in megabytes, one entry per worker.
    """

    output_tensor: torch.Tensor
    node_latencies_ms: list
    node_peak_memory_mb: list


class ResultStore:
    """
    Thread-safe slot map for correlating pipeline requests to results.

    The orchestrator creates a slot before firing RunShard, then waits
    on it. The PipelineCallbackServicer delivers the result when the
    last worker calls DeliverResult.
    """

    def __init__(self) -> None:
        self._slots: dict = {}
        self._lock = threading.Lock()

    def create_slot(self, request_id: str) -> threading.Event:
        """
        Reserve a slot for the given request_id.

        Parameters
        ----------
        request_id : str
            Unique identifier for the pipeline request.

        Returns
        -------
        threading.Event
            The event that fires when the result is delivered.
        """
        event = threading.Event()
        with self._lock:
            self._slots[request_id] = (event, None)
        return event

    def deliver(self, request_id: str, result: PipelineResult) -> bool:
        """
        Store the result and signal the waiting thread.

        Parameters
        ----------
        request_id : str
            The request to resolve.
        result : PipelineResult
            The result to store.

        Returns
        -------
        bool
            True if the slot existed, False if it was not found (e.g.
            already timed out).
        """
        with self._lock:
            if request_id not in self._slots:
                return False
            event, _ = self._slots[request_id]
            self._slots[request_id] = (event, result)
        event.set()
        return True

    def wait(self, request_id: str, timeout_s: float) -> PipelineResult:
        """
        Block until the result is delivered or the timeout expires.

        Parameters
        ----------
        request_id : str
            The request to wait for.
        timeout_s : float
            Maximum seconds to wait.

        Returns
        -------
        PipelineResult
            The delivered result.

        Raises
        ------
        TimeoutError
            If no result arrives before timeout_s.
        """
        with self._lock:
            slot = self._slots.get(request_id)
        if slot is None:
            raise KeyError(f"No slot for request_id '{request_id}'")

        event = slot[0]
        fired = event.wait(timeout=timeout_s)

        with self._lock:
            _, result = self._slots.pop(request_id, (None, None))

        if not fired:
            raise TimeoutError(f"Timed out waiting for pipeline result '{request_id}'")
        return result


class PipelineCallbackServicer(inference_pb2_grpc.PipelineCallbackServiceServicer):
    """
    gRPC handler for PipelineCallbackService on the orchestrator.

    Workers call DeliverResult when they finish the last shard. This
    servicer deserializes the output tensor and delivers it to the
    shared ResultStore so waiting PipelineRunner calls unblock.

    Parameters
    ----------
    result_store : ResultStore
        The shared slot map used by PipelineRunner.
    """

    def __init__(self, result_store: ResultStore) -> None:
        self._result_store = result_store

    def DeliverResult(self, request, context):
        """
        Receive a pipeline result from the last worker.

        Parameters
        ----------
        request : inference_pb2.DeliverResultRequest
            Contains request_id, serialized output tensor, and accumulated
            per-node timing and memory data.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.DeliverResultResponse
            Acknowledgement with success=True.
        """
        output_tensor = deserialize_tensor(
            request.output_tensor,
            list(request.output_shape),
            request.output_dtype,
        )
        result = PipelineResult(
            output_tensor=output_tensor,
            node_latencies_ms=list(request.node_latencies_ms),
            node_peak_memory_mb=list(request.node_peak_memory_mb),
        )
        self._result_store.deliver(request.request_id, result)
        return inference_pb2.DeliverResultResponse(acknowledged=True)


class PipelineRunner:
    """
    Orchestrates shard loading, pipeline execution, and shard unloading.

    load() extracts shard bytes from ModelStore and sends LoadShard to
    each worker in plan order. run_forward() fires RunShard at the first
    worker and blocks until the last worker calls DeliverResult. unload()
    sends UnloadShard to all workers.

    Parameters
    ----------
    model_store : ModelStore
        The orchestrator's full-model host for shard extraction.
    plan : DispatchPlan
        The dispatch plan describing which worker owns which layers.
    orchestrator_callback_address : str
        Address of the orchestrator's PipelineCallbackService (host:port).
        Passed to every worker so the last one knows where to deliver.
    result_store : ResultStore
        Shared slot map to register and wait for pipeline results.
    timeout_s : float
        Seconds to wait for a result before raising TimeoutError.
    """

    def __init__(
        self,
        model_store,
        plan,
        orchestrator_callback_address: str,
        result_store: ResultStore,
        timeout_s: float = 30.0,
    ) -> None:
        self._model_store = model_store
        self._plan = plan
        self._callback_address = orchestrator_callback_address
        self._result_store = result_store
        self._timeout_s = timeout_s

    def load(self) -> None:
        """
        Extract shard bytes and send LoadShard to each worker in order.
        """
        for assignment in self._plan.assignments:
            weights_bytes, config_json = self._model_store.extract_shard(
                self._plan.arch,
                assignment.layer_start,
                assignment.layer_end,
                assignment.is_first,
                assignment.is_last,
            )
            request = inference_pb2.LoadShardRequest(
                model_name=self._plan.model_name,
                arch=self._plan.arch,
                layer_start=assignment.layer_start,
                layer_end=assignment.layer_end,
                total_layers=self._plan.total_layers,
                next_worker_address=assignment.next_worker_address,
                shard_weights=weights_bytes,
                model_config_json=config_json,
            )
            addr = f"{assignment.ip_address}:{assignment.inference_port}"
            with WorkerClient(addr) as client:
                client.load_shard(request)

    def run_forward(self, input_tensor: torch.Tensor) -> PipelineResult:
        """
        Fire a pipeline forward pass and wait for the result.

        Assigns a UUID request_id, registers a result slot, fires RunShard
        at the first worker, and blocks until DeliverResult resolves the slot.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Token IDs to pass to the first shard.

        Returns
        -------
        PipelineResult
            The result delivered by the last worker.

        Raises
        ------
        TimeoutError
            If no result arrives within timeout_s.
        """
        request_id = str(uuid.uuid4())
        self._result_store.create_slot(request_id)

        out_data, out_shape, out_dtype = serialize_tensor(input_tensor)
        first = self._plan.assignments[0]
        run_request = inference_pb2.RunShardRequest(
            model_name=self._plan.model_name,
            input_tensor=out_data,
            input_shape=out_shape,
            input_dtype=out_dtype,
            request_id=request_id,
            orchestrator_callback_address=self._callback_address,
        )
        addr = f"{first.ip_address}:{first.inference_port}"
        with WorkerClient(addr) as client:
            client.run_shard(run_request)

        return self._result_store.wait(request_id, self._timeout_s)

    def unload(self) -> None:
        """
        Send UnloadShard to every worker in the plan.
        """
        for assignment in self._plan.assignments:
            request = inference_pb2.UnloadShardRequest(
                model_name=self._plan.model_name,
            )
            addr = f"{assignment.ip_address}:{assignment.inference_port}"
            with WorkerClient(addr) as client:
                client.unload_shard(request)
