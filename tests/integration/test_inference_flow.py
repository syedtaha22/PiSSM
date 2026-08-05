"""
Integration tests for the inference gRPC flow.

These tests spin up a real gRPC server with InferenceServiceServicer,
load the actual Mamba-130M model via gRPC, run inference, and verify
results. Marked @pytest.mark.slow because they download and load
the real model.

TestCachedGenerationRoundtrip below is the exception: it uses the
tiny, local, random-weight dummy-mamba-tiny checkpoint (see
scripts/make_dummy_manifest.py) over a real two-node gRPC pipeline,
so it stays in the fast suite - no model download, no @pytest.mark.slow.
"""

from concurrent import futures

import grpc
import pytest
import torch

from proto.generated import inference_pb2
from proto.generated import inference_pb2_grpc
from inference.manifest import ModelManifest
from inference.service import InferenceServiceServicer
from inference.tensor_utils import serialize_tensor
from orchestrator.dispatch import DispatchPlan, ShardAssignment
from orchestrator.model_store import ModelStore
from orchestrator.pipeline import PipelineCallbackServicer, PipelineRunner
from orchestrator.worker_client import _CHANNEL_OPTIONS


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


DUMMY_MANIFEST = ModelManifest(
    name="dummy-mamba-tiny",
    arch="mamba",
    checkpoint="checkpoints/dummy-mamba-tiny",
    layers=4,
    hidden_dim=64,
    state_dim=8,
    input_type="text",
    tokenizer="EleutherAI/gpt-neox-20b",
)


def _start_worker():
    """
    Start an in-process gRPC server hosting a real InferenceServiceServicer.

    Returns
    -------
    tuple[grpc.Server, int]
        The running server and the port it bound to.
    """
    servicer = InferenceServiceServicer()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4), options=_CHANNEL_OPTIONS
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port


def _start_callback_server(result_store):
    """
    Start an in-process gRPC server hosting PipelineCallbackServicer.

    Parameters
    ----------
    result_store : ResultStore
        The store the servicer delivers results into.

    Returns
    -------
    tuple[grpc.Server, int]
        The running server and the port it bound to.
    """
    servicer = PipelineCallbackServicer(result_store)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4), options=_CHANNEL_OPTIONS
    )
    inference_pb2_grpc.add_PipelineCallbackServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port


def _make_two_node_dispatch_plan(port0: int, port1: int) -> DispatchPlan:
    """
    Build a 2-node DispatchPlan for dummy-mamba-tiny's 4 layers, split
    evenly, pointing at two local gRPC ports.

    Parameters
    ----------
    port0 : int
        Port of the first (is_first) worker.
    port1 : int
        Port of the second (is_last) worker.

    Returns
    -------
    DispatchPlan
        A plan with node-0 owning layers [0,2) and node-1 owning [2,4).
    """
    return DispatchPlan(
        assignments=[
            ShardAssignment(
                node_id="node-0",
                ip_address="localhost",
                inference_port=port0,
                layer_start=0,
                layer_end=2,
                is_first=True,
                is_last=False,
                next_worker_address=f"localhost:{port1}",
            ),
            ShardAssignment(
                node_id="node-1",
                ip_address="localhost",
                inference_port=port1,
                layer_start=2,
                layer_end=4,
                is_first=False,
                is_last=True,
                next_worker_address="",
            ),
        ],
        arch="mamba",
        model_name="dummy-mamba-tiny",
        total_layers=4,
    )


class TestCachedGenerationRoundtrip:
    """
    Verifies recurrent-state caching over a real two-node gRPC pipeline.

    Uses the tiny local dummy-mamba-tiny checkpoint (no download) to
    drive two real InferenceServiceServicer workers and a real
    PipelineCallbackServicer, exercising the actual wire protocol
    (reset_cache field, WorkerClient, ResultStore) end to end - not
    just the in-process shard/service-layer parity already covered by
    unit tests. Asserts logit-level closeness per generated token
    rather than exact-text equality: since dummy-mamba-tiny is
    deterministic (eval mode, do_sample=False-equivalent greedy argmax,
    no dropout), the cached and full-recompute paths are expected to
    match near-exactly at every step, but comparing at the logit level
    is still the correct assertion per the project's floating-point
    drift caveat for autoregressive comparisons.
    """

    def test_cached_generate_matches_full_recompute_per_step(self, result_store):
        """
        runner.generate()'s cached per-token logits match a from-scratch
        full-recompute reference at every step, over a real gRPC pipeline.
        """
        worker_servers = []
        callback_server = None
        model_store = None
        try:
            port0_server, port0 = _start_worker()
            port1_server, port1 = _start_worker()
            worker_servers.extend([port0_server, port1_server])
            callback_server, callback_port = _start_callback_server(result_store)

            plan = _make_two_node_dispatch_plan(port0, port1)
            model_store = ModelStore()
            model_store.load(DUMMY_MANIFEST)

            runner = PipelineRunner(
                model_store=model_store,
                plan=plan,
                orchestrator_callback_address=f"localhost:{callback_port}",
                result_store=result_store,
                timeout_s=30.0,
            )
            runner.load()

            prompt = torch.tensor([[10, 20, 30]], dtype=torch.int64)
            max_new_tokens = 3

            cached = runner.generate(prompt, max_new_tokens)
            assert len(cached.step_results) == max_new_tokens

            reference_logits = []
            cur = prompt
            for _ in range(max_new_tokens):
                result = runner.run_forward(cur)
                reference_logits.append(result.output_tensor[0, -1, :])
                next_token = (
                    result.output_tensor[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                )
                cur = torch.cat([cur, next_token], dim=1)

            for step in range(max_new_tokens):
                cached_logits = cached.step_results[step].output_tensor[0, -1, :]
                assert torch.allclose(
                    cached_logits, reference_logits[step], atol=1e-4
                ), f"step {step}: cached and full-recompute logits diverged"

            expected_ids = prompt.clone()
            for logits in reference_logits:
                next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
                expected_ids = torch.cat([expected_ids, next_token], dim=1)
            assert torch.equal(cached.output_ids, expected_ids)
        finally:
            for server in worker_servers:
                server.stop(grace=2)
            if callback_server is not None:
                callback_server.stop(grace=2)
            if model_store is not None:
                model_store.unload()
