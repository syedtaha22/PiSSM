"""
Tests for WorkerClient and PipelineCallbackClient.

Both clients wrap gRPC stubs with thin method delegators and a context
manager that closes the channel. Tests mock the channel and stub
constructors to avoid real network connections.
"""

from unittest.mock import MagicMock, patch

from proto.generated import inference_pb2
from orchestrator.worker_client import PipelineCallbackClient, WorkerClient

_512MB = 512 * 1024 * 1024


class TestWorkerClient:
    """
    Tests for WorkerClient, which wraps InferenceServiceStub.
    """

    def test_load_shard_delegates_to_stub(self):
        """
        load_shard calls stub.LoadShard with the provided request.
        """
        with (
            patch("orchestrator.worker_client.grpc.insecure_channel"),
            patch("orchestrator.worker_client.InferenceServiceStub") as mock_stub_cls,
        ):
            mock_stub = MagicMock()
            mock_stub_cls.return_value = mock_stub
            client = WorkerClient("localhost:50052")

            request = inference_pb2.LoadShardRequest(model_name="mamba-130m")
            client.load_shard(request)

            mock_stub.LoadShard.assert_called_once_with(request)

    def test_run_shard_delegates_to_stub(self):
        """
        run_shard calls stub.RunShard with the provided request.
        """
        with (
            patch("orchestrator.worker_client.grpc.insecure_channel"),
            patch("orchestrator.worker_client.InferenceServiceStub") as mock_stub_cls,
        ):
            mock_stub = MagicMock()
            mock_stub_cls.return_value = mock_stub
            client = WorkerClient("localhost:50052")

            request = inference_pb2.RunShardRequest(model_name="mamba-130m")
            client.run_shard(request)

            mock_stub.RunShard.assert_called_once_with(request)

    def test_unload_shard_delegates_to_stub(self):
        """
        unload_shard calls stub.UnloadShard with the provided request.
        """
        with (
            patch("orchestrator.worker_client.grpc.insecure_channel"),
            patch("orchestrator.worker_client.InferenceServiceStub") as mock_stub_cls,
        ):
            mock_stub = MagicMock()
            mock_stub_cls.return_value = mock_stub
            client = WorkerClient("localhost:50052")

            request = inference_pb2.UnloadShardRequest(model_name="mamba-130m")
            client.unload_shard(request)

            mock_stub.UnloadShard.assert_called_once_with(request)

    def test_context_manager_closes_channel(self):
        """
        Exiting the context manager closes the gRPC channel exactly once.
        """
        mock_channel = MagicMock()
        with (
            patch(
                "orchestrator.worker_client.grpc.insecure_channel",
                return_value=mock_channel,
            ),
            patch("orchestrator.worker_client.InferenceServiceStub"),
        ):
            with WorkerClient("localhost:50052"):
                pass

        mock_channel.close.assert_called_once()

    def test_channel_configured_with_large_message_size(self):
        """
        The channel is created with max_send and max_receive options of
        at least 512 MB so that shard weight bytes fit in a single message.
        """
        with (
            patch(
                "orchestrator.worker_client.grpc.insecure_channel"
            ) as mock_channel_fn,
            patch("orchestrator.worker_client.InferenceServiceStub"),
        ):
            WorkerClient("localhost:50052")

            _, kwargs = mock_channel_fn.call_args
            options = dict(kwargs.get("options", []))
            assert options.get("grpc.max_send_message_length", 0) >= _512MB
            assert options.get("grpc.max_receive_message_length", 0) >= _512MB


class TestPipelineCallbackClient:
    """
    Tests for PipelineCallbackClient, which wraps PipelineCallbackServiceStub.
    """

    def test_deliver_result_delegates_to_stub(self):
        """
        deliver_result calls stub.DeliverResult with the provided request.
        """
        with (
            patch("orchestrator.worker_client.grpc.insecure_channel"),
            patch(
                "orchestrator.worker_client.PipelineCallbackServiceStub"
            ) as mock_stub_cls,
        ):
            mock_stub = MagicMock()
            mock_stub_cls.return_value = mock_stub
            client = PipelineCallbackClient("localhost:50060")

            request = inference_pb2.DeliverResultRequest(request_id="req-abc")
            client.deliver_result(request)

            mock_stub.DeliverResult.assert_called_once_with(request)
