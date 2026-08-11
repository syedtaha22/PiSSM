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
    GenerationStep,
    PipelineCallbackServicer,
    PipelineError,
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
        checkpoint="state-spaces/mamba-130m-hf",
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
    mock_store.extract_shard.return_value = ([], b"fake_config")
    runner = PipelineRunner(
        model_store=mock_store,
        plan=plan,
        orchestrator_callback_address=callback_address,
        result_store=result_store,
        timeout_s=timeout_s,
        load_timeout_s=timeout_s,
    )
    return runner, result_store


def make_shard_ready_request(
    model_name="mamba-130m",
    node_id="node-0",
    success=True,
    error_message="",
    memory_used_mb=12,
    layers_loaded=12,
):
    """
    Build a ShardReadyRequest with sensible defaults.

    Returns
    -------
    inference_pb2.ShardReadyRequest
        A populated shard-ready report.
    """
    return inference_pb2.ShardReadyRequest(
        model_name=model_name,
        node_id=node_id,
        success=success,
        error_message=error_message,
        memory_used_mb=memory_used_mb,
        layers_loaded=layers_loaded,
    )


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
    Tests for PipelineCallbackServicer: DeliverResult and
    ReportShardReady RPC handlers.
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

    def test_report_shard_ready_resolves_slot(self):
        """
        ReportShardReady delivers the reported outcome to the
        namespaced "shard:{model_name}:{node_id}" slot in ResultStore.
        """
        store = ResultStore()
        store.create_slot("shard:mamba-130m:node-0")
        servicer = PipelineCallbackServicer(store)

        request = make_shard_ready_request(
            model_name="mamba-130m",
            node_id="node-0",
            success=True,
            memory_used_mb=260,
            layers_loaded=12,
        )

        response = servicer.ReportShardReady(request, MagicMock())

        assert response.acknowledged is True
        result = store.wait("shard:mamba-130m:node-0", timeout_s=0.1)
        assert result.success is True
        assert result.memory_used_mb == 260
        assert result.layers_loaded == 12


# ---------------------------------------------------------------------------
# PipelineRunner.load
# ---------------------------------------------------------------------------


class TestPipelineRunnerLoad:
    """
    Tests for PipelineRunner.load: two-phase dispatch (send LoadShard
    metadata to every worker) then wait (block on every worker's
    ReportShardReady before returning).
    """

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_dispatches_load_shard_to_each_worker(self, mock_worker_cls):
        """
        load() calls WorkerClient.load_shard once for every assignment,
        and only returns once every worker's ReportShardReady arrives.
        """
        runner, result_store = make_runner()
        mock_client = mock_worker_cls.return_value.__enter__.return_value

        def deliver(request):
            is_node0 = request.layer_start == 0
            node_id = "node-0" if is_node0 else "node-1"
            servicer = PipelineCallbackServicer(result_store)
            servicer.ReportShardReady(
                make_shard_ready_request(model_name="mamba-130m", node_id=node_id),
                MagicMock(),
            )
            return inference_pb2.LoadShardResponse(success=True)

        mock_client.load_shard.side_effect = deliver

        runner.load()

        assert mock_client.load_shard.call_count == 2

    @patch("orchestrator.pipeline.tqdm")
    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_progress_bars_use_leave_false(self, mock_worker_cls, mock_tqdm):
        """
        Both of load()'s progress bars (dispatch and wait phases) are
        created with leave=False, so they clear themselves on
        completion instead of littering the terminal - this project's
        established progress-bar convention.
        """
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        runner, result_store = make_runner()
        mock_client = mock_worker_cls.return_value.__enter__.return_value

        def deliver(request):
            node_id = "node-0" if request.layer_start == 0 else "node-1"
            PipelineCallbackServicer(result_store).ReportShardReady(
                make_shard_ready_request(model_name="mamba-130m", node_id=node_id),
                MagicMock(),
            )
            return inference_pb2.LoadShardResponse(success=True)

        mock_client.load_shard.side_effect = deliver

        runner.load()

        assert mock_tqdm.call_count == 2
        for call in mock_tqdm.call_args_list:
            assert call.kwargs.get("leave") is False

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_sends_correct_next_worker_address_to_first_worker(
        self, mock_worker_cls
    ):
        """
        The first worker's LoadShard carries the second worker's address,
        checkpoint, and orchestrator callback address.
        """
        runner, result_store = make_runner()
        mock_client = mock_worker_cls.return_value.__enter__.return_value

        def deliver(request):
            node_id = "node-0" if request.layer_start == 0 else "node-1"
            PipelineCallbackServicer(result_store).ReportShardReady(
                make_shard_ready_request(model_name="mamba-130m", node_id=node_id),
                MagicMock(),
            )
            return inference_pb2.LoadShardResponse(success=True)

        mock_client.load_shard.side_effect = deliver

        runner.load()

        first_request = mock_client.load_shard.call_args_list[0][0][0]
        assert first_request.next_worker_address == "192.168.1.11:50052"
        assert first_request.checkpoint == "state-spaces/mamba-130m-hf"
        assert first_request.orchestrator_callback_address == "localhost:50060"

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_sends_empty_next_worker_address_to_last_worker(self, mock_worker_cls):
        """
        The last worker's LoadShard has an empty next_worker_address.
        """
        runner, result_store = make_runner()
        mock_client = mock_worker_cls.return_value.__enter__.return_value

        def deliver(request):
            node_id = "node-0" if request.layer_start == 0 else "node-1"
            PipelineCallbackServicer(result_store).ReportShardReady(
                make_shard_ready_request(model_name="mamba-130m", node_id=node_id),
                MagicMock(),
            )
            return inference_pb2.LoadShardResponse(success=True)

        mock_client.load_shard.side_effect = deliver

        runner.load()

        last_request = mock_client.load_shard.call_args_list[-1][0][0]
        assert last_request.next_worker_address == ""

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_raises_if_worker_rejects_load_shard(self, mock_worker_cls):
        """
        load() raises PipelineError if a worker's LoadShardResponse
        itself reports success=False (request rejected outright).
        """
        runner, _ = make_runner()
        mock_client = mock_worker_cls.return_value.__enter__.return_value
        mock_client.load_shard.return_value = inference_pb2.LoadShardResponse(
            success=False, error_message="model already loaded"
        )

        with pytest.raises(PipelineError, match="already loaded"):
            runner.load()

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_raises_if_worker_reports_shard_ready_failure(self, mock_worker_cls):
        """
        load() raises PipelineError if a worker's background load fails,
        reported via ReportShardReady after LoadShard itself succeeded.
        """
        runner, result_store = make_runner()
        mock_client = mock_worker_cls.return_value.__enter__.return_value

        def deliver(request):
            node_id = "node-0" if request.layer_start == 0 else "node-1"
            success = node_id != "node-1"
            PipelineCallbackServicer(result_store).ReportShardReady(
                make_shard_ready_request(
                    model_name="mamba-130m",
                    node_id=node_id,
                    success=success,
                    error_message="" if success else "checkpoint fetch failed",
                ),
                MagicMock(),
            )
            return inference_pb2.LoadShardResponse(success=True)

        mock_client.load_shard.side_effect = deliver

        with pytest.raises(PipelineError, match="checkpoint fetch failed"):
            runner.load()

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_raises_on_load_wait_timeout(self, mock_worker_cls):
        """
        load() dispatches LoadShard to every worker (the dispatch phase
        completes even though nothing will ever finish "downloading"),
        then raises TimeoutError waiting for a ReportShardReady that
        never arrives - proving dispatch and wait are separate phases.
        """
        runner, _ = make_runner(timeout_s=0.01)
        mock_client = mock_worker_cls.return_value.__enter__.return_value
        mock_client.load_shard.return_value = inference_pb2.LoadShardResponse(
            success=True
        )

        with pytest.raises(TimeoutError):
            runner.load()

        assert mock_client.load_shard.call_count == 2

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_timeout_defaults_to_waiting_indefinitely(self, mock_worker_cls):
        """
        PipelineRunner's load_timeout_s defaults to None (wait forever)
        when not given explicitly - a shard download has no natural
        deadline to guess at, unlike a single forward pass. Verified by
        checking the value ResultStore.wait actually receives, not by
        waiting forever.
        """
        plan = make_two_node_plan()
        mock_result_store = MagicMock()
        mock_result_store.wait.return_value = MagicMock(success=True)
        mock_store = MagicMock()
        mock_store.extract_shard.return_value = ([], b"fake_config")
        runner = PipelineRunner(
            model_store=mock_store,
            plan=plan,
            orchestrator_callback_address="localhost:50060",
            result_store=mock_result_store,
        )
        mock_client = mock_worker_cls.return_value.__enter__.return_value
        mock_client.load_shard.return_value = inference_pb2.LoadShardResponse(
            success=True
        )

        runner.load()

        for call in mock_result_store.wait.call_args_list:
            assert call.args[1] is None

    @patch("orchestrator.pipeline.WorkerClient")
    def test_load_waits_on_workers_concurrently_not_sequentially(self, mock_worker_cls):
        """
        A worker that reports failure quickly is not held up by another
        worker that never reports at all - load() must wait on every
        worker's ReportShardReady concurrently, not one at a time in
        assignment order. If it waited sequentially on node-0 (which
        never delivers) before ever checking node-1 (which fails fast),
        this would take the full load_timeout_s instead of ~0.1s.
        """
        runner, result_store = make_runner(timeout_s=5.0)
        mock_client = mock_worker_cls.return_value.__enter__.return_value

        def deliver(request):
            # node-0 (layer_start=0) never reports ready. node-1
            # (layer_start=12) reports a failure almost immediately.
            if request.layer_start == 12:

                def report_late():
                    PipelineCallbackServicer(result_store).ReportShardReady(
                        make_shard_ready_request(
                            model_name="mamba-130m",
                            node_id="node-1",
                            success=False,
                            error_message="checkpoint fetch failed",
                        ),
                        MagicMock(),
                    )

                threading.Timer(0.05, report_late).start()
            return inference_pb2.LoadShardResponse(success=True)

        mock_client.load_shard.side_effect = deliver

        start = time.monotonic()
        with pytest.raises(PipelineError, match="checkpoint fetch failed"):
            runner.load()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, (
            f"load() took {elapsed:.2f}s - looks like it waited on node-0 "
            "before ever checking node-1's already-known failure"
        )


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
# PipelineRunner.generate_stream
# ---------------------------------------------------------------------------


class TestPipelineRunnerGenerateStream:
    """
    Tests for PipelineRunner.generate_stream(): the generator form of
    generate() that yields one GenerationStep per token as it's produced,
    so callers (e.g. a streaming HTTP endpoint) can forward tokens to a
    client incrementally instead of waiting for the whole generation.
    """

    @patch("orchestrator.pipeline.WorkerClient")
    def test_yields_one_step_per_token(self, mock_worker_cls):
        """
        Consuming the generator fully yields exactly max_new_tokens
        GenerationStep objects, each carrying a token_id and the
        PipelineResult that produced it.
        """
        runner, result_store = make_runner()
        next_tokens = [111, 222, 333]

        def deliver(request):
            step = (
                len(
                    [
                        c
                        for c in mock_worker_cls.return_value.__enter__.return_value.run_shard.call_args_list
                    ]
                )
                - 1
            )
            result_store.deliver(
                request.request_id, make_argmax_result(next_tokens[step])
            )
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        steps = list(runner.generate_stream(prompt, max_new_tokens=3))

        assert len(steps) == 3
        assert all(isinstance(s, GenerationStep) for s in steps)
        assert [s.token_id.item() for s in steps] == next_tokens
        assert all(isinstance(s.step_result, PipelineResult) for s in steps)

    @patch("orchestrator.pipeline.WorkerClient")
    def test_first_step_resets_cache_subsequent_do_not(self, mock_worker_cls):
        """
        generate_stream() drives the same reset_cache protocol as
        generate(): reset_cache=True with the full prompt on the first
        step, reset_cache=False with just the newest token afterward.
        """
        runner, result_store = make_runner()
        requests = []
        next_tokens = [111, 222]

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
        list(runner.generate_stream(prompt, max_new_tokens=2))

        assert requests[0].reset_cache is True
        assert requests[1].reset_cache is False

    @patch("orchestrator.pipeline.WorkerClient")
    def test_lock_held_for_entire_iteration_not_just_first_step(self, mock_worker_cls):
        """
        The _generation_lock stays held for as long as the caller keeps
        iterating, not just until the first token is yielded - an
        overlapping generate_stream() (or generate()) call against the
        same runner must still be serialized.
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

        def consume():
            list(runner.generate_stream(prompt, max_new_tokens=2))

        threads = [threading.Thread(target=consume) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert concurrency["max_active"] == 1

    @patch("orchestrator.pipeline.WorkerClient")
    def test_generate_matches_generate_stream_output(self, mock_worker_cls):
        """
        generate() (built on top of generate_stream()) still returns the
        same output_ids/step_results shape as before the refactor - a
        regression guard for the internal-sharing change.
        """
        runner, result_store = make_runner()
        next_tokens = [111, 222]

        def deliver(request):
            step = (
                len(
                    [
                        c
                        for c in mock_worker_cls.return_value.__enter__.return_value.run_shard.call_args_list
                    ]
                )
                - 1
            )
            result_store.deliver(
                request.request_id, make_argmax_result(next_tokens[step])
            )
            return MagicMock()

        mock_worker_cls.return_value.__enter__.return_value.run_shard.side_effect = (
            deliver
        )

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        result = runner.generate(prompt, max_new_tokens=2)

        assert torch.equal(
            result.output_ids, torch.tensor([[1, 2, 3, 111, 222]], dtype=torch.int64)
        )
        assert len(result.step_results) == 2


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
