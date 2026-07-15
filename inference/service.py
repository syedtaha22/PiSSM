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
import psutil
import torch

from proto.generated import inference_pb2
from proto.generated import inference_pb2_grpc
from inference.loader import (
    ModelHandle,
    load_model,
    load_shard_from_bytes,
    unload_model,
)
from inference.manifest import ModelManifest
from inference.tensor_utils import deserialize_tensor, serialize_tensor
from orchestrator.worker_client import PipelineCallbackClient, WorkerClient

logger = logging.getLogger(__name__)


class InferenceServiceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    Handles LoadShard, RunShard, and UnloadShard RPCs on worker nodes.

    Manages loaded models in an internal dictionary keyed by model name.
    Thread-safe for concurrent gRPC handler access.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelHandle] = {}
        self._lock = threading.Lock()

    def LoadShard(self, request, context):
        """
        Load a model into worker memory.

        Constructs a ModelManifest from the request fields, calls the
        loader, and stores the resulting ModelHandle.

        Parameters
        ----------
        request : inference_pb2.LoadShardRequest
            The load request from the orchestrator.
        context : grpc.ServicerContext
            The gRPC call context.

        Returns
        -------
        inference_pb2.LoadShardResponse
            Success status, memory used, and layers loaded.
        """
        with self._lock:
            if request.model_name in self._models:
                return inference_pb2.LoadShardResponse(
                    success=False,
                    error_message=f"Model '{request.model_name}' is already loaded",
                )

        try:
            if request.shard_weights:
                is_first = request.layer_start == 0
                is_last = request.layer_end == request.total_layers
                handle = load_shard_from_bytes(
                    shard_weights_bytes=request.shard_weights,
                    model_config_json_bytes=request.model_config_json,
                    arch=request.arch,
                    layer_start=request.layer_start,
                    layer_end=request.layer_end,
                    is_first=is_first,
                    is_last=is_last,
                    next_worker_address=request.next_worker_address,
                    model_name=request.model_name,
                )
                layers_loaded = request.layer_end - request.layer_start
            else:
                manifest = ModelManifest(
                    name=request.model_name,
                    arch=request.arch,
                    checkpoint=request.checkpoint,
                    layers=0,
                    hidden_dim=0,
                    state_dim=0,
                    input_type="text",
                    tokenizer=request.tokenizer,
                )
                handle = load_model(manifest)
                layers_loaded = handle.manifest.layers

            with self._lock:
                self._models[request.model_name] = handle

            logger.info(
                "Loaded model '%s': ~%d MB",
                request.model_name,
                handle.memory_mb,
            )

            return inference_pb2.LoadShardResponse(
                success=True,
                memory_used_mb=handle.memory_mb,
                layers_loaded=layers_loaded,
            )

        except Exception as e:
            logger.error("Failed to load model '%s': %s", request.model_name, e)
            return inference_pb2.LoadShardResponse(
                success=False,
                error_message=str(e),
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
                with torch.no_grad():
                    output_tensor = handle.model(input_tensor)
            elif request.generate_mode:
                with torch.no_grad():
                    output_tensor = handle.model.generate(
                        input_tensor,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=False,
                    )
            else:
                with torch.no_grad():
                    model_output = handle.model(input_tensor)
                    output_tensor = model_output.logits

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
            logger.error("Inference failed on '%s': %s", request.model_name, e)
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
