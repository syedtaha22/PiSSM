"""
Tests for the pipeline coordinator.

Covers ResultStore (thread-safe slot map), PipelineCallbackServicer
(gRPC handler), and PipelineRunner (load/run/unload orchestration).
All WorkerClient calls are mocked; no real network I/O occurs.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch

from inference.tensor_utils import deserialize_tensor, serialize_tensor
from orchestrator.dispatch import DispatchPlan, ShardAssignment
from orchestrator.pipeline import (
    GenerationResult,
    PipelineCallbackServicer,
    PipelineResult,
    PipelineRunner,
    ResultStore,
)
from proto.generated import inference_pb2


def make_two_node_plan():
    """
    Return a DispatchPlan with two nodes splitting 24 layers evenly.

    Returns
    -------
    DispatchPlan
        Plan with node-0 (is_first=True) and node-1 (is_last=True).
    """
    return DispatchPlan(
        assignments=[
            ShardAssignment(
                node_id="node-0",
                ip_address="192.168.1.10",
                inference_port=50052,
                layer_start=0,
                layer_end=12,
                is_first=True,
                is_last=False,
                next_worker_address="192.168.1.11:50052",
            ),
            ShardAssignment(
                node_id="node-1",
                ip_address="192.168.1.11",
                inference_port=50052,
                layer_start=12,
                layer_end=24,
                is_first=False,
                is_last=True,
                next_worker_address="",
            ),
        ],
        arch="mamba",
        model_name="mamba-130m",
        total_layers=24,
    )


def make_runner(plan=None, callback_address="localhost:50060", timeout_s=1.0):
    """
    Create a PipelineRunner with a fresh ResultStore and mocked ModelStore.

    Parameters
    ----------
    plan : DispatchPlan or None
        Dispatch plan to use. Defaults to make_two_node_plan().
    callback_address : str
        Orchestrator callback address passed to workers.
    timeout_s : float
        Wait timeout for run_forward.

    Returns
    -------
    tuple[PipelineRunner, ResultStore]
        The runner and the underlying result store.
    """
    if plan is None:
        plan = make_two_node_plan()
    result_store = ResultStore()
    mock_store = MagicMock()
    mock_store.extract_shard.return_value = (b"fake_weights", b"fake_config")
    runner = PipelineRunner(
        model_store=mock_store,
        plan=plan,
        orchestrator_callback_address=callback_address,
        result_store=result_store,
        timeout_s=timeout_s,
    )
    return runner, result_store


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------


class TestResultStore:
    """
    Tests for ResultStore: thread-safe slot creation, delivery, and wait.
    """

    def test_wait_returns_delivered_result(self):
        """
        Delivering a result before or during wait returns it from wait.
        """
        store = ResultStore()
        expected = PipelineResult(
            output_tensor=torch.randn(1, 3, 50280),
            node_latencies_ms=[10.0, 12.0],
            node_peak_memory_mb=[256, 300],
        )

        store.create_slot("req-1")

        def deliver():
            store.deliver("req-1", expected)

        t = threading.Thread(target=deliver)
        t.start()
        result = store.wait("req-1", timeout_s=1.0)
        t.join()

        assert result is expected

    def test_wait_raises_on_timeout(self):
        """
        wait raises TimeoutError when no result is delivered before timeout_s.
        """
        store = ResultStore()
        store.create_slot("req-timeout")

        with pytest.raises(TimeoutError, match="req-timeout"):
            store.wait("req-timeout", timeout_s=0.01)


# ---------------------------------------------------------------------------
# PipelineCallbackServicer
# ---------------------------------------------------------------------------


class TestPipelineCallbackServicer:
    """
    Tests for PipelineCallbackServicer: DeliverResult RPC handler.
    """

    def test_deliver_result_resolves_slot(self):
        """
        DeliverResult deserializes the tensor and delivers to ResultStore.
        """

        store = ResultStore()
        store.create_slot("req-42")
        servicer = PipelineCallbackServicer(store)

        output = torch.randn(1, 3, 50280)
        data, shape, dtype_str = serialize_tensor(output)

        request = inference_pb2.DeliverResultRequest(
            request_id="req-42",
            output_tensor=data,
            output_shape=shape,
            output_dtype=dtype_str,
            node_latencies_ms=[8.5, 11.2],
            node_peak_memory_mb=[240, 310],
        )

        response = servicer.DeliverResult(request, MagicMock())

        assert response.acknowledged is True
        result = store.wait("req-42", timeout_s=0.1)
        assert isinstance(result, PipelineResult)
        assert result.output_tensor.shape == output.shape
        assert result.node_latencies_ms == pytest.approx([8.5, 11.2])
        assert result.node_peak_memory_mb == [240, 310]


# ---------------------------------------------------------------------------
# PipelineRunner.load
# ---------------------------------------------------------------------------


class TestPipelineRunnerLoad:
    """
    Tests for PipelineRunner.load: sends LoadShard to each worker.
    """

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_sends_load_shard_to_each_worker(self, mock_worker_cls):
        """
        load() calls WorkerClient.load_shard once for every assignment.
        """
        runner, _ = make_runner()
        runner.load()

        mock_client = mock_worker_cls.return_value.__enter__.return_value
        assert mock_client.load_shard.call_count == 2

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_sends_correct_next_worker_address_to_first_worker(
        self, mock_worker_cls
    ):
        """
        The first worker's LoadShard carries the second worker's address.
        """
        runner, _ = make_runner()
        runner.load()

        mock_client = mock_worker_cls.return_value.__enter__.return_value
        first_request = mock_client.load_shard.call_args_list[0][0][0]
        assert first_request.next_worker_address == "192.168.1.11:50052"

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_sends_empty_next_worker_address_to_last_worker(self, mock_worker_cls):
        """
        The last worker's LoadShard has an empty next_worker_address.
        """
        runner, _ = make_runner()
        runner.load()

        mock_client = mock_worker_cls.return_value.__enter__.return_value
        last_request = mock_client.load_shard.call_args_list[-1][0][0]
        assert last_request.next_worker_address == ""


# ---------------------------------------------------------------------------
# PipelineRunner.run_forward
# ---------------------------------------------------------------------------


class TestPipelineRunnerRunForward:
    """
    Tests for PipelineRunner.run_forward: fire-and-collect pipeline execution.
    """

    @patch("orchestrator.pipeline.WorkerClient")
    def test_run_forward_sends_run_shard_only_to_first_worker(self, mock_worker_cls):
        """
        run_forward fires RunShard only at the first worker's address.
        """
        runner, result_store = make_runner()

        def deliver(request):
            result_store.deliver(
                request.request_id,
                PipelineResult(
                    output_tensor=torch.randn(1, 3, 50280),
                    node_latencies_ms=[10.0],
                    node_peak_memory_mb=[256],
                ),
            )
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        runner.run_forward(torch.tensor([[101, 2023]], dtype=torch.int64))

        mock_worker_cls.assert_called_once_with("192.168.1.10:50052")

    @patch("orchestrator.pipeline.WorkerClient")
    def test_run_forward_includes_request_id_and_callback_address(
        self, mock_worker_cls
    ):
        """
        The RunShard request carries a non-empty request_id and the
        orchestrator's callback address.
        """
        runner, result_store = make_runner(callback_address="localhost:50060")

        def deliver(request):
            result_store.deliver(
                request.request_id,
                PipelineResult(
                    output_tensor=torch.randn(1, 3, 50280),
                    node_latencies_ms=[10.0],
                    node_peak_memory_mb=[256],
                ),
            )
            return MagicMock()

        mock_client = mock_worker_cls.return_value.__enter__.return_value
        mock_client.run_shard.side_effect = deliver

        runner.run_forward(torch.tensor([[101, 2023]], dtype=torch.int64))

        run_request = mock_client.run_shard.call_args[0][0]
        assert run_request.request_id != ""
        assert run_request.orchestrator_callback_address == "localhost:50060"

    @patch("orchestrator.pipeline.WorkerClient")
    def test_run_forward_returns_pipeline_result_after_future_resolves(
        self, mock_worker_cls
    ):
        """
        run_forward returns the PipelineResult once the slot is delivered.
        """
        runner, result_store = make_runner()
        expected_tensor = torch.randn(1, 3, 50280)

        def deliver(request):
            result_store.deliver(
                request.request_id,
                PipelineResult(
                    output_tensor=expected_tensor,
                    node_latencies_ms=[10.0, 12.0],
                    node_peak_memory_mb=[256, 300],
                ),
            )
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        result = runner.run_forward(torch.tensor([[101, 2023]], dtype=torch.int64))

        assert isinstance(result, PipelineResult)
        assert result.output_tensor is expected_tensor
        assert result.node_latencies_ms == pytest.approx([10.0, 12.0])

    @patch("orchestrator.pipeline.WorkerClient")
    def test_run_forward_raises_on_timeout(self, mock_worker_cls):
        """
        run_forward raises TimeoutError if no DeliverResult arrives in time.
        """
        runner, _ = make_runner(timeout_s=0.01)

        with pytest.raises(TimeoutError):
            runner.run_forward(torch.tensor([[101, 2023]], dtype=torch.int64))


# ---------------------------------------------------------------------------
# PipelineRunner.generate
# ---------------------------------------------------------------------------


def make_argmax_result(next_token_id, vocab=50280, seq_len=1):
    """
    Build a PipelineResult whose output_tensor argmaxes to next_token_id
    at its last position, for driving mocked generate() steps.

    Parameters
    ----------
    next_token_id : int
        The token id that should win the argmax at the last position.
    vocab : int
        Vocabulary size for the fake logits tensor.
    seq_len : int
        Sequence length of the fake logits tensor.

    Returns
    -------
    PipelineResult
        A result with node_latencies_ms=[5.0], node_peak_memory_mb=[200].
    """
    logits = torch.full((1, seq_len, vocab), -1.0)
    logits[0, -1, next_token_id] = 10.0
    return PipelineResult(
        output_tensor=logits,
        node_latencies_ms=[5.0],
        node_peak_memory_mb=[200],
    )


class TestPipelineRunnerGenerate:
    """
    Tests for PipelineRunner.generate(): cached autoregressive decoding
    that sends reset_cache=True with the full prompt on the first step,
    then reset_cache=False with only the newest token on every
    subsequent step.
    """

    @patch("orchestrator.pipeline.WorkerClient")
    def test_first_step_resets_cache_and_sends_full_prompt(self, mock_worker_cls):
        """
        Step 1's RunShardRequest has reset_cache=True and carries the
        entire prompt, not just the newest token.
        """
        runner, result_store = make_runner()
        requests = []
        next_tokens = [111, 222, 333]

        def deliver(request):
            requests.append(request)
            step = len(requests) - 1
            result_store.deliver(
                request.request_id, make_argmax_result(next_tokens[step])
            )
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        runner.generate(prompt, max_new_tokens=3)

        assert requests[0].reset_cache is True
        first_input = deserialize_tensor(
            requests[0].input_tensor,
            list(requests[0].input_shape),
            requests[0].input_dtype,
        )
        assert torch.equal(first_input, prompt)

    @patch("orchestrator.pipeline.WorkerClient")
    def test_subsequent_steps_reuse_cache_and_send_only_newest_token(
        self, mock_worker_cls
    ):
        """
        Steps after the first have reset_cache=False and carry only the
        single newest token, not the growing sequence.
        """
        runner, result_store = make_runner()
        requests = []
        next_tokens = [111, 222, 333]

        def deliver(request):
            requests.append(request)
            step = len(requests) - 1
            result_store.deliver(
                request.request_id, make_argmax_result(next_tokens[step])
            )
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        runner.generate(prompt, max_new_tokens=3)

        for step, expected_token in enumerate(next_tokens[:-1], start=1):
            assert requests[step].reset_cache is False
            step_input = deserialize_tensor(
                requests[step].input_tensor,
                list(requests[step].input_shape),
                requests[step].input_dtype,
            )
            assert step_input.shape == (1, 1)
            assert step_input.item() == expected_token

    @patch("orchestrator.pipeline.WorkerClient")
    def test_returns_output_ids_and_one_step_result_per_token(self, mock_worker_cls):
        """
        generate() returns a GenerationResult whose output_ids concatenates
        the prompt with every generated token, and whose step_results has
        exactly one PipelineResult per generated token.
        """
        runner, result_store = make_runner()
        next_tokens = [111, 222]

        def deliver(request):
            step = deliver.calls
            deliver.calls += 1
            result_store.deliver(
                request.request_id, make_argmax_result(next_tokens[step])
            )
            return MagicMock()

        deliver.calls = 0
        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        result = runner.generate(prompt, max_new_tokens=2)

        assert isinstance(result, GenerationResult)
        assert torch.equal(
            result.output_ids, torch.tensor([[1, 2, 3, 111, 222]], dtype=torch.int64)
        )
        assert len(result.step_results) == 2
        assert all(isinstance(r, PipelineResult) for r in result.step_results)

    @patch("orchestrator.pipeline.WorkerClient")
    def test_concurrent_generate_calls_are_serialized(self, mock_worker_cls):
        """
        Two overlapping generate() calls against the same runner never
        run concurrently - the lock must hold for the entire call, since
        interleaving would corrupt worker-side cache state.
        """
        runner, result_store = make_runner()
        concurrency = {"active": 0, "max_active": 0}
        state_lock = threading.Lock()

        def deliver(request):
            with state_lock:
                concurrency["active"] += 1
                concurrency["max_active"] = max(
                    concurrency["max_active"], concurrency["active"]
                )
            time.sleep(0.05)
            result_store.deliver(request.request_id, make_argmax_result(1))
            with state_lock:
                concurrency["active"] -= 1
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        prompt = torch.tensor([[1, 2]], dtype=torch.int64)
        threads = [
            threading.Thread(target=runner.generate, args=(prompt, 2)) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert concurrency["max_active"] == 1


# ---------------------------------------------------------------------------
# PipelineRunner.unload
# ---------------------------------------------------------------------------


class TestPipelineRunnerUnload:
    """
    Tests for PipelineRunner.unload: sends UnloadShard to all workers.
    """

    @patch("orchestrator.pipeline.WorkerClient")
    def test_unload_sends_unload_shard_to_all_workers(self, mock_worker_cls):
        """
        unload() calls WorkerClient.unload_shard once for every assignment.
        """
        runner, _ = make_runner()
        runner.unload()

        mock_client = mock_worker_cls.return_value.__enter__.return_value
        assert mock_client.unload_shard.call_count == 2

    @patch("orchestrator.pipeline.WorkerClient")
    def test_unload_continues_past_an_unreachable_worker(self, mock_worker_cls):
        """
        A dead node (e.g. crashed, or replaced before redistributing)
        doesn't stop the remaining nodes from being unloaded.
        """
        runner, _ = make_runner()

        mock_client = mock_worker_cls.return_value.__enter__.return_value
        dead_node_error = grpc.RpcError()
        dead_node_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        dead_node_error.details = lambda: "Connection refused"
        mock_client.unload_shard.side_effect = [dead_node_error, None]

        runner.unload()

        assert mock_client.unload_shard.call_count == 2
