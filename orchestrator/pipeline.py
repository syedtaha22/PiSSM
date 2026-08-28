"""
Pipeline coordinator for circular pipeline-parallel inference.

ResultStore holds per-request futures that the last worker resolves via
DeliverResult, and (under a "shard:{model_name}:{node_id}" namespaced
key) per-shard load outcomes that a worker resolves via ReportShardReady.
PipelineCallbackServicer is the gRPC handler that bridges both incoming
RPCs to the ResultStore. PipelineRunner orchestrates LoadShard (dispatch
metadata to every worker, then wait for every worker's ReportShardReady),
the initial RunShard fire-and-forward, and UnloadShard across all
workers in a DispatchPlan.
"""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import grpc
import torch
from tqdm import tqdm

from inference.tensor_utils import deserialize_tensor, serialize_tensor
from orchestrator.worker_client import WorkerClient
from proto.generated import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """
    Raised when a worker rejects or fails a pipeline load/run request.
    """


@dataclass
class ShardReadyResult:
    """
    Outcome of a single worker's background LoadShard download-and-build.

    Parameters
    ----------
    success : bool
        True if the shard was built and loaded successfully.
    error_message : str
        Error description if not success, empty otherwise.
    memory_used_mb : int
        Approximate memory consumed by the loaded shard.
    layers_loaded : int
        Number of model layers loaded.
    """

    success: bool
    error_message: str
    memory_used_mb: int
    layers_loaded: int


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


@dataclass
class GenerationResult:
    """
    Result of a full autoregressive generation via PipelineRunner.generate().

    Parameters
    ----------
    output_ids : torch.Tensor
        Token ids of shape (1, prompt_len + max_new_tokens): the original
        prompt followed by every generated token.
    step_results : list[PipelineResult]
        One PipelineResult per generated token, in generation order.
    """

    output_ids: torch.Tensor
    step_results: list


@dataclass
class GenerationStep:
    """
    One generated token, produced by PipelineRunner.generate_stream().

    Parameters
    ----------
    token_id : torch.Tensor
        The newly generated token id, shape (1, 1).
    step_result : PipelineResult
        The pipeline result that produced this token.
    """

    token_id: torch.Tensor
    step_result: PipelineResult


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

    def ReportShardReady(self, request, context):
        """
        Receive a shard-load outcome from a worker's background load.

        Parameters
        ----------
        request : inference_pb2.ShardReadyRequest
            Carries model_name, node_id, and the load outcome.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.ShardReadyResponse
            Acknowledgement with acknowledged=True.
        """
        result = ShardReadyResult(
            success=request.success,
            error_message=request.error_message,
            memory_used_mb=request.memory_used_mb,
            layers_loaded=request.layers_loaded,
        )
        self._result_store.deliver(
            f"shard:{request.model_name}:{request.node_id}", result
        )
        return inference_pb2.ShardReadyResponse(acknowledged=True)


class PipelineRunner:
    """
    Orchestrates shard loading, pipeline execution, and shard unloading.

    load() resolves each shard's tensor metadata from ModelStore and
    dispatches it to every worker via LoadShard, then waits for every
    worker's ReportShardReady before returning - two separate phases,
    so every worker fetches its own weights independently and in
    parallel with the others. run_forward() fires RunShard at the first
    worker and blocks until the last worker calls DeliverResult.
    unload() sends UnloadShard to all workers.

    Parameters
    ----------
    model_store : ModelStore
        The orchestrator's checkpoint-metadata host for shard planning.
    plan : DispatchPlan
        The dispatch plan describing which worker owns which layers.
    orchestrator_callback_address : str
        Address of the orchestrator's PipelineCallbackService (host:port).
        Passed to every worker so it knows where to call back
        ReportShardReady (and, for the last shard, DeliverResult).
    result_store : ResultStore
        Shared slot map to register and wait for pipeline results and
        shard-load outcomes.
    timeout_s : float
        Seconds to wait for an inference result before raising
        TimeoutError.
    load_timeout_s : float or None
        Seconds to wait for a worker's ReportShardReady before raising
        TimeoutError. None (the default) waits indefinitely; pass a
        number to bound it.
    """

    def __init__(
        self,
        model_store,
        plan,
        orchestrator_callback_address: str,
        result_store: ResultStore,
        timeout_s: float = 30.0,
        load_timeout_s: float | None = None,
    ) -> None:
        self._model_store = model_store
        self._plan = plan
        self._callback_address = orchestrator_callback_address
        self._result_store = result_store
        self._timeout_s = timeout_s
        self._load_timeout_s = load_timeout_s
        self._generation_lock = threading.Lock()

    def _shard_ready_key(self, node_id: str) -> str:
        """
        Build the ResultStore key a worker's ReportShardReady resolves.

        Parameters
        ----------
        node_id : str
            The worker's node ID.

        Returns
        -------
        str
            Namespaced key unique to this model and node.
        """
        return f"shard:{self._plan.model_name}:{node_id}"

    def load(self) -> None:
        """
        Dispatch shard metadata to every worker, then wait for all to load.

        Two phases: first sends LoadShard (tensor locations and config)
        to every worker in plan order, so each one starts fetching its
        own weights independently as soon as it's dispatched. Only once
        every worker has been dispatched does this wait for each one's
        ReportShardReady, so a slow download on one worker doesn't
        delay starting the next worker's download.

        Raises
        ------
        PipelineError
            If a worker rejects LoadShard outright, or reports a failed
            load via ReportShardReady.
        TimeoutError
            If a worker's ReportShardReady doesn't arrive within
            load_timeout_s.
        """
        for assignment in self._plan.assignments:
            self._result_store.create_slot(self._shard_ready_key(assignment.node_id))

        n = len(self._plan.assignments)
        with tqdm(
            total=n, desc="dispatching shard metadata", unit="shard", leave=False
        ) as pbar:
            for assignment in self._plan.assignments:
                addr = f"{assignment.ip_address}:{assignment.inference_port}"
                pbar.set_description(
                    f"shard [{assignment.layer_start},{assignment.layer_end}) -> {addr}"
                )
                tensor_locations, config_json = self._model_store.extract_shard(
                    self._plan.arch,
                    assignment.layer_start,
                    assignment.layer_end,
                    assignment.is_first,
                    assignment.is_last,
                )
                request = inference_pb2.LoadShardRequest(
                    model_name=self._plan.model_name,
                    checkpoint=self._plan.checkpoint,
                    arch=self._plan.arch,
                    layer_start=assignment.layer_start,
                    layer_end=assignment.layer_end,
                    total_layers=self._plan.total_layers,
                    next_worker_address=assignment.next_worker_address,
                    tensor_locations=[
                        inference_pb2.TensorLocation(
                            dest_name=location.dest_name,
                            source_name=location.source_name,
                            source_file=location.source_file,
                        )
                        for location in tensor_locations
                    ],
                    model_config_json=config_json,
                    orchestrator_callback_address=self._callback_address,
                )
                with WorkerClient(addr) as client:
                    response = client.load_shard(request)
                if not response.success:
                    raise PipelineError(
                        f"worker {assignment.node_id} rejected LoadShard: "
                        f"{response.error_message}"
                    )
                pbar.update(1)

        executor = ThreadPoolExecutor(max_workers=n)
        try:
            futures = {
                executor.submit(
                    self._result_store.wait,
                    self._shard_ready_key(assignment.node_id),
                    self._load_timeout_s,
                ): assignment
                for assignment in self._plan.assignments
            }
            with tqdm(
                total=n,
                desc="waiting for workers to finish loading",
                unit="shard",
                leave=False,
            ) as pbar:
                for future in as_completed(futures):
                    assignment = futures[future]
                    result = future.result()
                    if not result.success:
                        raise PipelineError(
                            f"worker {assignment.node_id} failed to load shard: "
                            f"{result.error_message}"
                        )
                    pbar.set_description(f"{assignment.node_id} ready")
                    pbar.update(1)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def run_forward(self, input_tensor: torch.Tensor) -> PipelineResult:
        """
        Fire a single, standalone pipeline forward pass and wait for the result.

        Always starts a fresh generation (reset_cache=True): this is a
        one-off pass, not a step in an autoregressive decode via generate(),
        so it must never reuse recurrent-state cache left over from a
        prior generate() call against the same resident model.

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
        return self._run_shard_step(input_tensor, reset_cache=True)

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> GenerationResult:
        """
        Run one full autoregressive generation using per-request recurrent
        cache state.

        Built on top of generate_stream(), consuming it fully rather than
        duplicating its reset_cache/lock logic - see that method for the
        per-step protocol.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized prompt of shape (1, prompt_len).
        max_new_tokens : int
            Number of tokens to generate.

        Returns
        -------
        GenerationResult
            The full output_ids and one PipelineResult per generated token.

        Raises
        ------
        TimeoutError
            If no result arrives within timeout_s for any step.
        """
        output_ids = input_ids
        step_results = []
        for step in self.generate_stream(input_ids, max_new_tokens):
            output_ids = torch.cat([output_ids, step.token_id], dim=1)
            step_results.append(step.step_result)
        return GenerationResult(output_ids=output_ids, step_results=step_results)

    def generate_stream(self, input_ids: torch.Tensor, max_new_tokens: int):
        """
        Generator form of generate(): yields one GenerationStep per token
        as it's produced, instead of returning only after the whole
        generation completes.

        Lets a caller (e.g. a streaming HTTP endpoint) forward each token
        to a client incrementally. Sends exactly one RunShard per new
        token: the first call has reset_cache=True and carries the full
        prompt (prefill, rebuilds every worker's cache); each subsequent
        call has reset_cache=False and carries only the single newest
        token (decode, reuses cache). Holds self._generation_lock for as
        long as the caller keeps iterating - not just until the first
        token is yielded - so an overlapping generate()/generate_stream()
        call against the same resident model is serialized rather than
        corrupting worker-side cache state.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized prompt of shape (1, prompt_len).
        max_new_tokens : int
            Number of tokens to generate.

        Yields
        ------
        GenerationStep
            One per generated token, in generation order.

        Raises
        ------
        TimeoutError
            If no result arrives within timeout_s for any step.
        """
        with self._generation_lock:
            next_input = input_ids
            for step in range(max_new_tokens):
                result = self._run_shard_step(next_input, reset_cache=(step == 0))
                next_token = (
                    result.output_tensor[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                )
                next_input = next_token
                yield GenerationStep(token_id=next_token, step_result=result)

    def _run_shard_step(
        self, input_tensor: torch.Tensor, reset_cache: bool
    ) -> PipelineResult:
        """
        Fire one RunShardRequest at the first worker and wait for the result.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Token IDs (first shard) to send for this step.
        reset_cache : bool
            True to have every worker discard its cache and rebuild fresh
            before running input_tensor; False to reuse existing cache.

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
            reset_cache=reset_cache,
        )
        addr = f"{first.ip_address}:{first.inference_port}"
        with WorkerClient(addr) as client:
            client.run_shard(run_request)

        return self._result_store.wait(request_id, self._timeout_s)

    def unload(self) -> None:
        """
        Send UnloadShard to every worker in the plan.

        A worker that's unreachable (e.g. it crashed or was replaced)
        is logged and skipped rather than aborting the rest - callers
        such as redistribution rely on every reachable node getting
        freed even if one node in the old plan is gone.
        """
        for assignment in self._plan.assignments:
            request = inference_pb2.UnloadShardRequest(
                model_name=self._plan.model_name,
            )
            addr = f"{assignment.ip_address}:{assignment.inference_port}"
            try:
                with WorkerClient(addr) as client:
                    client.unload_shard(request)
            except grpc.RpcError as err:
                logger.warning(
                    "Failed to unload '%s' on %s: %s: %s",
                    self._plan.model_name,
                    addr,
                    err.code().name,
                    err.details(),
                )
