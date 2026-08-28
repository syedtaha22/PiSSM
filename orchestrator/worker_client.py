"""
gRPC client wrappers for inter-node communication in PiSSM.

WorkerClient wraps the InferenceService stub. The orchestrator uses it
to send LoadShard, RunShard, and UnloadShard to workers. Workers also
use it to forward RunShard to the next node in the circular pipeline.

PipelineCallbackClient wraps the PipelineCallbackService stub. The last
worker in a pipeline uses it to deliver inference results directly back
to the orchestrator without routing through intermediate nodes. Every
worker uses it to report a LoadShard outcome via ReportShardReady once
its background load finishes.

Both clients configure their channels with a 256 MB message size limit.
"""

import grpc

from proto.generated.inference_pb2_grpc import (
    InferenceServiceStub,
    PipelineCallbackServiceStub,
)

_MAX_MESSAGE_BYTES = 256 * 1024 * 1024

_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
]


class WorkerClient:
    """
    Client for the InferenceService on a worker node.

    Wraps LoadShard, RunShard, and UnloadShard with a thin delegating
    interface. Supports the context manager protocol; the channel is
    closed on exit.

    Parameters
    ----------
    address : str
        Address of the target worker's InferenceService (host:port).
    """

    def __init__(self, address: str) -> None:
        self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTIONS)
        self._stub = InferenceServiceStub(self._channel)

    def __enter__(self):
        """
        Enter the context manager.

        Returns
        -------
        WorkerClient
            Self.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        Close the gRPC channel.
        """
        self._channel.close()

    def load_shard(self, request):
        """
        Call LoadShard on the remote worker.

        Parameters
        ----------
        request : inference_pb2.LoadShardRequest
            The load request to send.

        Returns
        -------
        inference_pb2.LoadShardResponse
            Response from the worker.
        """
        return self._stub.LoadShard(request)

    def run_shard(self, request):
        """
        Call RunShard on the remote worker.

        Parameters
        ----------
        request : inference_pb2.RunShardRequest
            The run request to send.

        Returns
        -------
        inference_pb2.RunShardResponse
            Acknowledgement from the worker.
        """
        return self._stub.RunShard(request)

    def unload_shard(self, request):
        """
        Call UnloadShard on the remote worker.

        Parameters
        ----------
        request : inference_pb2.UnloadShardRequest
            The unload request to send.

        Returns
        -------
        inference_pb2.UnloadShardResponse
            Response from the worker.
        """
        return self._stub.UnloadShard(request)


class PipelineCallbackClient:
    """
    Client for the PipelineCallbackService on the orchestrator.

    Used by the last worker in a pipeline to deliver inference results
    directly back to the orchestrator, completing the circular topology.
    Supports the context manager protocol; the channel is closed on exit.

    Parameters
    ----------
    address : str
        Address of the orchestrator's PipelineCallbackService (host:port).
    """

    def __init__(self, address: str) -> None:
        self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTIONS)
        self._stub = PipelineCallbackServiceStub(self._channel)

    def __enter__(self):
        """
        Enter the context manager.

        Returns
        -------
        PipelineCallbackClient
            Self.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        Close the gRPC channel.
        """
        self._channel.close()

    def deliver_result(self, request):
        """
        Call DeliverResult on the orchestrator.

        Parameters
        ----------
        request : inference_pb2.DeliverResultRequest
            The result delivery request carrying the output tensor and
            accumulated per-node timing and memory data.

        Returns
        -------
        inference_pb2.DeliverResultResponse
            Acknowledgement from the orchestrator.
        """
        return self._stub.DeliverResult(request)

    def report_shard_ready(self, request, timeout_s: float = 10.0):
        """
        Call ReportShardReady on the orchestrator.

        This runs on a worker's background load thread, with nothing
        else supervising it - an explicit timeout is required so a
        slow-to-resolve or unreachable orchestrator address can never
        block that thread forever.

        Parameters
        ----------
        request : inference_pb2.ShardReadyRequest
            Carries the outcome of a worker's background LoadShard
            download-and-build: success/failure, error message, memory
            used, and layers loaded.
        timeout_s : float
            Maximum seconds to wait for the call to complete.

        Returns
        -------
        inference_pb2.ShardReadyResponse
            Acknowledgement from the orchestrator.
        """
        return self._stub.ReportShardReady(request, timeout=timeout_s)
