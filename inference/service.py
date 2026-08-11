"""
gRPC servicer implementation for the InferenceService.

Handles model loading, inference execution, and model unloading on
worker nodes. The orchestrator calls these RPCs to instruct workers
to load models, run forward passes or generation, and release models
from memory.
"""

import logging
import threading
import time

import grpc
import httpx
import psutil
import torch

from proto.generated import inference_pb2
from proto.generated import inference_pb2_grpc
from inference.loader import ModelHandle, load_shard_from_metadata, unload_model
from inference.tensor_utils import deserialize_tensor, serialize_tensor
from orchestrator.worker_client import PipelineCallbackClient, WorkerClient

logger = logging.getLogger(__name__)


def _format_error(e: Exception) -> str:
    """
    Render an exception as a single log line.

    grpc.RpcError's default str() spans several lines (status, details,
    debug string). huggingface_hub's HTTP errors (raised when a
    checkpoint can't be resolved, e.g. a bad repo id or a 401/404) span
    even more - status line, blank line, explanation, a "please make
    sure..." hint, and a docs link. Both are compacted to one line here
    so callers can log or send this over the wire without flooding
    either.
    """
    if isinstance(e, grpc.RpcError) and hasattr(e, "code") and hasattr(e, "details"):
        return f"{e.code()}: {e.details()}"
    response = getattr(e, "response", None)
    if isinstance(e, httpx.HTTPError) and response is not None:
        return (
            f"HTTP {response.status_code} {response.reason_phrase} for {response.url}"
        )
    return str(e)


class InferenceServiceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    Handles LoadShard, RunShard, and UnloadShard RPCs on worker nodes.

    Manages loaded models in an internal dictionary keyed by model name.
    Thread-safe for concurrent gRPC handler access.

    Parameters
    ----------
    node_id : str
        This worker's node ID, sent back in every ReportShardReady call
        so the orchestrator can tell which dispatch-plan assignment a
        report corresponds to (several workers may load the same
        model_name concurrently).
    """

    def __init__(self, node_id: str = "") -> None:
        self._node_id = node_id
        self._models: dict[str, ModelHandle] = {}
        self._loading: set[str] = set()
        self._lock = threading.Lock()

    def LoadShard(self, request, context):
        """
        Accept a shard-load request and start loading in the background.

        Returns as soon as the request is accepted - it does not wait
        for the shard to actually finish loading. The real load (fetch
        checkpoint file(s), build the shard, load its tensors) runs on
        a background thread, which reports its outcome to the
        orchestrator via PipelineCallbackService.ReportShardReady once
        done.

        Parameters
        ----------
        request : inference_pb2.LoadShardRequest
            The load request from the orchestrator.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.LoadShardResponse
            success=True means the request was accepted (background
            load started, or the model was already fully loaded).
            success=False only for a request rejected outright because
            this model is already loading (request is in flight).
        """
        with self._lock:
            if request.model_name in self._models:
                handle = self._models[request.model_name]
                threading.Thread(
                    target=self._report_shard_ready,
                    args=(request,),
                    kwargs={
                        "success": True,
                        "error_message": "",
                        "memory_used_mb": handle.memory_mb,
                        "layers_loaded": request.layer_end - request.layer_start,
                    },
                    daemon=True,
                ).start()
                return inference_pb2.LoadShardResponse(success=True)

            if request.model_name in self._loading:
                return inference_pb2.LoadShardResponse(
                    success=False,
                    error_message=f"Model '{request.model_name}' is already loaded",
                )
            self._loading.add(request.model_name)

        threading.Thread(
            target=self._load_shard_background, args=(request,), daemon=True
        ).start()

        return inference_pb2.LoadShardResponse(success=True)

    def _load_shard_background(self, request) -> None:
        """
        Build and load a shard, then report the outcome to the orchestrator.

        Runs on a background thread started by LoadShard. On success,
        stores the resulting ModelHandle in self._models. Either way,
        calls back ReportShardReady on the orchestrator's
        PipelineCallbackService with the outcome.

        Parameters
        ----------
        request : inference_pb2.LoadShardRequest
            The load request that triggered this background load.
        """
        logger.info(
            "Loading shard '%s': layers [%d, %d) from '%s'",
            request.model_name,
            request.layer_start,
            request.layer_end,
            request.checkpoint,
        )
        try:
            is_first = request.layer_start == 0
            is_last = request.layer_end == request.total_layers
            handle = load_shard_from_metadata(
                tensor_locations=list(request.tensor_locations),
                model_config_json_bytes=request.model_config_json,
                arch=request.arch,
                layer_start=request.layer_start,
                layer_end=request.layer_end,
                is_first=is_first,
                is_last=is_last,
                checkpoint=request.checkpoint,
                next_worker_address=request.next_worker_address,
                model_name=request.model_name,
            )
            with self._lock:
                self._models[request.model_name] = handle
                self._loading.discard(request.model_name)

            logger.info(
                "Loaded model '%s': ~%d MB", request.model_name, handle.memory_mb
            )
            self._report_shard_ready(
                request,
                success=True,
                error_message="",
                memory_used_mb=handle.memory_mb,
                layers_loaded=request.layer_end - request.layer_start,
            )

        except Exception as e:
            with self._lock:
                self._loading.discard(request.model_name)
            error_message = _format_error(e)
            logger.error(
                "Failed to load model '%s': %s", request.model_name, error_message
            )
            self._report_shard_ready(
                request,
                success=False,
                error_message=error_message,
                memory_used_mb=0,
                layers_loaded=0,
            )

    def _report_shard_ready(
        self,
        request,
        success: bool,
        error_message: str,
        memory_used_mb: int,
        layers_loaded: int,
    ) -> None:
        """
        Call ReportShardReady on the orchestrator with a load outcome.

        A failure to deliver this report (e.g. the orchestrator is
        briefly unreachable) is logged and swallowed rather than
        raised - there is no caller left to propagate it to on a
        background thread, and the orchestrator's own wait will simply
        time out.

        Parameters
        ----------
        request : inference_pb2.LoadShardRequest
            The load request this report corresponds to.
        success : bool
            Whether the shard loaded successfully.
        error_message : str
            Error description if not success, empty otherwise.
        memory_used_mb : int
            Approximate memory consumed by the loaded shard.
        layers_loaded : int
            Number of model layers loaded.
        """
        try:
            with PipelineCallbackClient(
                request.orchestrator_callback_address
            ) as client:
                client.report_shard_ready(
                    inference_pb2.ShardReadyRequest(
                        model_name=request.model_name,
                        node_id=self._node_id,
                        success=success,
                        error_message=error_message,
                        memory_used_mb=memory_used_mb,
                        layers_loaded=layers_loaded,
                    )
                )
        except grpc.RpcError as err:
            logger.error(
                "Failed to report shard ready for '%s' to %s: %s",
                request.model_name,
                request.orchestrator_callback_address,
                _format_error(err),
            )

    def RunShard(self, request, context):
        """
        Execute inference on a loaded model.

        Deserializes the input tensor, runs either a forward pass or
        autoregressive generation, serializes the output, and records
        latency and peak memory.

        Parameters
        ----------
        request : inference_pb2.RunShardRequest
            The run request from the orchestrator.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.RunShardResponse
            Success status, output tensor, latency, and peak memory.
        """
        with self._lock:
            handle = self._models.get(request.model_name)

        if handle is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{request.model_name}' is not loaded")
            return inference_pb2.RunShardResponse(
                success=False,
                error_message=f"Model '{request.model_name}' is not loaded",
            )

        try:
            input_tensor = deserialize_tensor(
                request.input_tensor,
                list(request.input_shape),
                request.input_dtype,
            )

            process = psutil.Process()
            start_time = time.perf_counter()
            pipeline_mode = bool(request.orchestrator_callback_address)

            if pipeline_mode:
                if request.reset_cache or handle.cache is None:
                    handle.cache = handle.model.new_cache()
                with torch.no_grad():
                    output_tensor = handle.model(
                        input_tensor, cache_params=handle.cache
                    )
            elif request.generate_mode:
                with torch.no_grad():
                    output_tensor = handle.model.generate(
                        input_tensor,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=False,
                    )
            else:
                # handle.model is a MambaShardModule (even a single,
                # whole-model "shard"), whose forward() returns the raw
                # logits tensor directly - no .logits attribute to read.
                with torch.no_grad():
                    output_tensor = handle.model(input_tensor)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            peak_memory_mb = process.memory_info().rss // (1024 * 1024)

            if pipeline_mode:
                accumulated_latencies = list(request.node_latencies_ms) + [elapsed_ms]
                accumulated_memory = list(request.node_peak_memory_mb) + [
                    peak_memory_mb
                ]
                out_data, out_shape, out_dtype = serialize_tensor(output_tensor)

                if handle.next_worker_address:
                    next_request = inference_pb2.RunShardRequest(
                        model_name=request.model_name,
                        input_tensor=out_data,
                        input_shape=out_shape,
                        input_dtype=out_dtype,
                        max_new_tokens=request.max_new_tokens,
                        request_id=request.request_id,
                        orchestrator_callback_address=request.orchestrator_callback_address,
                        node_latencies_ms=accumulated_latencies,
                        node_peak_memory_mb=accumulated_memory,
                        reset_cache=request.reset_cache,
                    )
                    with WorkerClient(handle.next_worker_address) as client:
                        client.run_shard(next_request)
                else:
                    deliver_request = inference_pb2.DeliverResultRequest(
                        request_id=request.request_id,
                        output_tensor=out_data,
                        output_shape=out_shape,
                        output_dtype=out_dtype,
                        node_latencies_ms=accumulated_latencies,
                        node_peak_memory_mb=accumulated_memory,
                    )
                    with PipelineCallbackClient(
                        request.orchestrator_callback_address
                    ) as client:
                        client.deliver_result(deliver_request)

                logger.debug(
                    "Pipeline shard '%s': %.1f ms, forwarded",
                    request.model_name,
                    elapsed_ms,
                )
                return inference_pb2.RunShardResponse(success=True)

            if isinstance(output_tensor, torch.Tensor):
                out_data, out_shape, out_dtype = serialize_tensor(output_tensor)
            else:
                out_data = b""
                out_shape = []
                out_dtype = ""

            logger.info(
                "Inference on '%s': %.1f ms, peak %d MB",
                request.model_name,
                elapsed_ms,
                peak_memory_mb,
            )

            return inference_pb2.RunShardResponse(
                success=True,
                output_tensor=out_data,
                output_shape=out_shape,
                output_dtype=out_dtype,
                latency_ms=elapsed_ms,
                peak_memory_mb=peak_memory_mb,
            )

        except Exception as e:
            logger.error(
                "Inference failed on '%s': %s", request.model_name, _format_error(e)
            )
            return inference_pb2.RunShardResponse(
                success=False,
                error_message=str(e),
            )

    def Ping(self, request, context):
        """
        Echo the request payload back unchanged.

        Used by benchmark_network.py to measure round-trip latency and
        throughput at multiple payload sizes.

        Parameters
        ----------
        request : inference_pb2.PingRequest
            Carries an arbitrary byte payload.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.PingResponse
            The same payload echoed back.
        """
        return inference_pb2.PingResponse(payload=request.payload)

    def UnloadShard(self, request, context):
        """
        Release a loaded model from worker memory.

        Parameters
        ----------
        request : inference_pb2.UnloadShardRequest
            The unload request from the orchestrator.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.UnloadShardResponse
            Success status and memory freed.
        """
        with self._lock:
            handle = self._models.pop(request.model_name, None)

        if handle is None:
            return inference_pb2.UnloadShardResponse(
                success=False,
                memory_freed_mb=0,
            )

        memory_freed = unload_model(handle)
        logger.info(
            "Unloaded model '%s': ~%d MB freed",
            request.model_name,
            memory_freed,
        )

        return inference_pb2.UnloadShardResponse(
            success=True,
            memory_freed_mb=memory_freed,
        )
