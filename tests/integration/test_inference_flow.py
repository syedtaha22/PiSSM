"""
Integration tests for the inference gRPC flow.

These tests spin up a real gRPC server with InferenceServiceServicer,
load the actual Mamba-130M model via gRPC, run inference, and verify
results. Marked @pytest.mark.slow because they download and load
the real model.
"""

from concurrent import futures

import grpc
import pytest
import torch

from proto.generated import inference_pb2
from proto.generated import inference_pb2_grpc
from inference.service import InferenceServiceServicer
from inference.tensor_utils import serialize_tensor


@pytest.fixture(scope="module")
def inference_server_and_channel():
    """
    Start an in-process gRPC server with InferenceServiceServicer.

    Yields
    ------
    tuple[grpc.Server, grpc.Channel, InferenceServiceServicer]
        The running server, a connected channel, and the servicer.
    """
    servicer = InferenceServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")

    yield server, channel, servicer

    channel.close()
    server.stop(grace=0)


def load_model_via_grpc(stub):
    """
    Load Mamba-130M via the gRPC LoadShard RPC.

    Parameters
    ----------
    stub : inference_pb2_grpc.InferenceServiceStub
        The gRPC stub.

    Returns
    -------
    inference_pb2.LoadShardResponse
        The load response.
    """
    return stub.LoadShard(
        inference_pb2.LoadShardRequest(
            model_name="mamba-130m",
            checkpoint="state-spaces/mamba-130m-hf",
            tokenizer="EleutherAI/gpt-neox-20b",
            arch="mamba",
            layer_start=0,
            layer_end=0,
        )
    )


@pytest.mark.slow
class TestInferenceRoundtrip:
    """
    Tests the full load-run-unload cycle over real gRPC.
    """

    def test_load_and_run(self, inference_server_and_channel):
        """
        Load the model, run inference, and get non-empty output.
        """
        _, channel, _ = inference_server_and_channel
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        load_resp = load_model_via_grpc(stub)
        assert load_resp.success is True

        input_ids = torch.tensor([[12764, 849, 403, 368, 2509, 32]], dtype=torch.int64)
        data, shape, dtype_str = serialize_tensor(input_ids)

        run_resp = stub.RunShard(
            inference_pb2.RunShardRequest(
                model_name="mamba-130m",
                input_tensor=data,
                input_shape=shape,
                input_dtype=dtype_str,
                max_new_tokens=10,
                generate_mode=True,
            )
        )

        assert run_resp.success is True
        assert run_resp.latency_ms > 0
        assert len(run_resp.output_tensor) > 0

    def test_run_without_load(self, inference_server_and_channel):
        """
        RunShard before LoadShard raises NOT_FOUND.
        """
        _, channel, _ = inference_server_and_channel
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        input_ids = torch.tensor([[101]], dtype=torch.int64)
        data, shape, dtype_str = serialize_tensor(input_ids)

        with pytest.raises(grpc.RpcError) as exc_info:
            stub.RunShard(
                inference_pb2.RunShardRequest(
                    model_name="nonexistent-model",
                    input_tensor=data,
                    input_shape=shape,
                    input_dtype=dtype_str,
                    max_new_tokens=5,
                    generate_mode=True,
                )
            )

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    def test_unload_after_load(self, inference_server_and_channel):
        """
        UnloadShard after LoadShard returns success and frees the model.
        """
        _, channel, servicer = inference_server_and_channel
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        if "unload-test-model" not in servicer._models:
            stub.LoadShard(
                inference_pb2.LoadShardRequest(
                    model_name="unload-test-model",
                    checkpoint="state-spaces/mamba-130m-hf",
                    tokenizer="EleutherAI/gpt-neox-20b",
                    arch="mamba",
                )
            )

        resp = stub.UnloadShard(
            inference_pb2.UnloadShardRequest(model_name="unload-test-model")
        )

        assert resp.success is True
        assert servicer._models.get("unload-test-model") is None
