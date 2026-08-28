"""
Tests for the InferenceServiceServicer gRPC handler.

Validates that the servicer correctly bridges gRPC requests to the
model loader. Tests call servicer methods directly with mock context
objects and a mocked loader, avoiding any model downloads or network I/O.
"""

import threading
from unittest.mock import MagicMock, patch

import grpc
import httpx
import torch
from transformers import MambaConfig, MambaForCausalLM

from inference.loader import ModelHandle
from inference.shard import MambaShardModule
from inference.tensor_utils import deserialize_tensor, serialize_tensor
from proto.generated import inference_pb2


def make_load_request(
    model_name="mamba-130m",
    checkpoint="state-spaces/mamba-130m-hf",
    arch="mamba",
    layer_start=0,
    layer_end=12,
    total_layers=24,
    next_worker_address="192.168.1.11:50052",
    orchestrator_callback_address="localhost:50060",
):
    """
    Build a LoadShardRequest with sensible defaults.

    tensor_locations is left empty since these tests mock
    load_shard_from_metadata directly rather than parsing it - only
    the request's other fields (checkpoint, layer range, callback
    address) matter at the service layer.

    Parameters
    ----------
    model_name : str
        Model name.
    checkpoint : str
        HuggingFace repo id or local checkpoint path.
    arch : str
        Architecture string.
    layer_start : int
        First layer index (inclusive).
    layer_end : int
        Last layer index (exclusive).
    total_layers : int
        Total layers in the full model.
    next_worker_address : str
        Address of the next worker in the pipeline, or empty string.
    orchestrator_callback_address : str
        Address of the orchestrator's PipelineCallbackService.

    Returns
    -------
    inference_pb2.LoadShardRequest
        A populated load request.
    """
    return inference_pb2.LoadShardRequest(
        model_name=model_name,
        checkpoint=checkpoint,
        arch=arch,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        next_worker_address=next_worker_address,
        orchestrator_callback_address=orchestrator_callback_address,
    )


def make_run_request(
    model_name="mamba-130m",
    input_tensor=None,
    max_new_tokens=10,
    generate_mode=True,
):
    """
    Build a RunShardRequest with sensible defaults.

    Parameters
    ----------
    model_name : str
        Model name.
    input_tensor : torch.Tensor or None
        Input tensor. Defaults to a small int64 tensor.
    max_new_tokens : int
        Max tokens for generation.
    generate_mode : bool
        Whether to use generation mode.

    Returns
    -------
    inference_pb2.RunShardRequest
        A populated run request.
    """
    if input_tensor is None:
        input_tensor = torch.tensor([[101, 2023, 3045]], dtype=torch.int64)
    data, shape, dtype_str = serialize_tensor(input_tensor)
    return inference_pb2.RunShardRequest(
        model_name=model_name,
        input_tensor=data,
        input_shape=shape,
        input_dtype=dtype_str,
        max_new_tokens=max_new_tokens,
        generate_mode=generate_mode,
    )


def make_mock_handle(name="mamba-130m", next_worker_address="", memory_mb=260):
    """
    Create a mock ModelHandle.

    Parameters
    ----------
    name : str
        Model name.
    next_worker_address : str
        Address of the next worker in the pipeline, or empty string.
    memory_mb : int
        Value reported as the handle's memory usage.

    Returns
    -------
    MagicMock
        A mock handle with expected attributes.
    """
    handle = MagicMock()
    handle.name = name
    handle.memory_mb = memory_mb
    handle.next_worker_address = next_worker_address
    return handle


def make_pipeline_run_request(
    model_name="mamba-130m",
    input_tensor=None,
    request_id="req-abc",
    orchestrator_callback_address="localhost:50060",
):
    """
    Build a RunShardRequest for pipeline mode.

    Parameters
    ----------
    model_name : str
        Model name.
    input_tensor : torch.Tensor or None
        Input tensor. Defaults to a small int64 tensor.
    request_id : str
        UUID-style correlation identifier.
    orchestrator_callback_address : str
        Address of the orchestrator's PipelineCallbackService.

    Returns
    -------
    inference_pb2.RunShardRequest
        A populated run request in pipeline mode.
    """
    if input_tensor is None:
        input_tensor = torch.tensor([[101, 2023, 3045]], dtype=torch.int64)
    data, shape, dtype_str = serialize_tensor(input_tensor)
    return inference_pb2.RunShardRequest(
        model_name=model_name,
        input_tensor=data,
        input_shape=shape,
        input_dtype=dtype_str,
        request_id=request_id,
        orchestrator_callback_address=orchestrator_callback_address,
    )


class TestLoadShard:
    """
    Tests for the LoadShard RPC's async accept-then-background-load flow.
    """

    @patch("inference.service.PipelineCallbackClient")
    @patch("inference.service.load_shard_from_metadata")
    def test_load_shard_returns_immediately_without_blocking(
        self, mock_load, mock_callback_cls
    ):
        """
        LoadShard returns success=True as soon as the request is
        accepted, without waiting for load_shard_from_metadata to
        finish - the actual load happens on a background thread.
        """
        from inference.service import InferenceServiceServicer

        unblock = threading.Event()

        def blocking_load(**kwargs):
            unblock.wait(timeout=2.0)
            return make_mock_handle()

        mock_load.side_effect = blocking_load
        servicer = InferenceServiceServicer()
        context = MagicMock()

        response = servicer.LoadShard(make_load_request(), context)

        assert response.success is True
        assert not unblock.is_set()  # proves the RPC didn't wait on the load
        unblock.set()

    @patch("inference.service.PipelineCallbackClient")
    @patch("inference.service.load_shard_from_metadata")
    def test_background_load_success_stores_model_and_reports_ready(
        self, mock_load, mock_callback_cls
    ):
        """
        A successful background load stores the handle in _models and
        reports success=True with its memory/layers to ReportShardReady.
        """
        from inference.service import InferenceServiceServicer

        mock_load.return_value = make_mock_handle(memory_mb=260)
        mock_client = mock_callback_cls.return_value.__enter__.return_value
        servicer = InferenceServiceServicer(node_id="node-0")

        servicer._load_shard_background(make_load_request(layer_start=0, layer_end=12))

        assert servicer._models.get("mamba-130m") is not None
        mock_client.report_shard_ready.assert_called_once()
        sent = mock_client.report_shard_ready.call_args[0][0]
        assert sent.model_name == "mamba-130m"
        assert sent.node_id == "node-0"
        assert sent.success is True
        assert sent.memory_used_mb == 260
        assert sent.layers_loaded == 12

    @patch("inference.service.PipelineCallbackClient")
    @patch("inference.service.load_shard_from_metadata")
    def test_background_load_failure_reports_not_ready(
        self, mock_load, mock_callback_cls
    ):
        """
        A failed background load does not store a model and reports
        success=False with the error message to ReportShardReady.
        """
        from inference.service import InferenceServiceServicer

        mock_load.side_effect = NotImplementedError("arch 's4' not supported")
        mock_client = mock_callback_cls.return_value.__enter__.return_value
        servicer = InferenceServiceServicer()

        servicer._load_shard_background(make_load_request(arch="s4"))

        assert servicer._models.get("mamba-130m") is None
        sent = mock_client.report_shard_ready.call_args[0][0]
        assert sent.success is False
        assert "s4" in sent.error_message

    @patch("inference.service.PipelineCallbackClient")
    def test_load_shard_already_loaded_reports_ready(self, mock_callback_cls):
        """
        LoadShard for a model already present in _models is not a
        failure - the model is loaded, which is the desired end state,
        so it reports success and still calls ReportShardReady for
        this request (using the already-resident handle's stats), so
        an orchestrator retrying a load it previously gave up on isn't
        left waiting forever on a report that will never come.
        """
        from inference.service import InferenceServiceServicer

        servicer = InferenceServiceServicer(node_id="node-0")
        handle = make_mock_handle(memory_mb=260)
        servicer._models["mamba-130m"] = handle
        context = MagicMock()
        mock_client = mock_callback_cls.return_value.__enter__.return_value

        response = servicer.LoadShard(
            make_load_request(layer_start=0, layer_end=12), context
        )

        assert response.success is True
        mock_client.report_shard_ready.assert_called_once()
        sent = mock_client.report_shard_ready.call_args[0][0]
        assert sent.model_name == "mamba-130m"
        assert sent.node_id == "node-0"
        assert sent.success is True
        assert sent.memory_used_mb == 260
        assert sent.layers_loaded == 12

    @patch("inference.service.PipelineCallbackClient")
    @patch("inference.service.load_shard_from_metadata")
    def test_background_load_http_failure_reports_compact_error(
        self, mock_load, mock_callback_cls
    ):
        """
        A background load failing with a multi-line HTTP error (e.g. a
        checkpoint HuggingFace can't resolve) reports a compact
        single-line error_message to ReportShardReady, not the raw
        multi-paragraph exception text.
        """
        from inference.service import InferenceServiceServicer

        response = MagicMock()
        response.status_code = 404
        response.reason_phrase = "Not Found"
        response.url = "https://huggingface.co/checkpoints/dummy-mamba-tiny/resolve/main/model.safetensors"
        err = httpx.HTTPError(
            "404 Client Error. (Request ID: ...)\n\nRepository Not Found for url: "
            "...\nPlease make sure you specified the correct `repo_id`."
        )
        err.response = response
        mock_load.side_effect = err
        mock_client = mock_callback_cls.return_value.__enter__.return_value
        servicer = InferenceServiceServicer()

        servicer._load_shard_background(make_load_request())

        sent = mock_client.report_shard_ready.call_args[0][0]
        assert sent.success is False
        assert "\n" not in sent.error_message
        assert sent.error_message == "HTTP 404 Not Found for " + response.url

    @patch("inference.service.PipelineCallbackClient")
    @patch("inference.service.load_shard_from_metadata")
    def test_load_shard_duplicate_in_flight_rejects(self, mock_load, mock_callback_cls):
        """
        A second LoadShard for a model whose first load is still
        in-flight is rejected synchronously rather than racing it.
        """
        from inference.service import InferenceServiceServicer

        unblock = threading.Event()

        def blocking_load(**kwargs):
            unblock.wait(timeout=2.0)
            return make_mock_handle()

        mock_load.side_effect = blocking_load
        servicer = InferenceServiceServicer()
        context = MagicMock()

        first = servicer.LoadShard(make_load_request(), context)
        second = servicer.LoadShard(make_load_request(), context)

        assert first.success is True
        assert second.success is False
        assert "already loaded" in second.error_message
        unblock.set()

    @patch("inference.service.PipelineCallbackClient")
    @patch("inference.service.load_shard_from_metadata")
    def test_background_load_logs_start_before_loading(
        self, mock_load, mock_callback_cls, caplog
    ):
        """
        The background thread logs that it started loading a shard
        before calling load_shard_from_metadata, so a worker's terminal
        shows immediate feedback instead of staying silent until the
        load finishes (successfully or not).
        """
        import logging

        from inference.service import InferenceServiceServicer

        mock_load.return_value = make_mock_handle()
        servicer = InferenceServiceServicer()

        with caplog.at_level(logging.INFO, logger="inference.service"):
            servicer._load_shard_background(
                make_load_request(model_name="mamba-130m", layer_start=0, layer_end=12)
            )

        assert len(caplog.records) >= 2, "expected a start log and a completion log"
        first_message = caplog.records[0].message
        assert "mamba-130m" in first_message
        assert "0" in first_message and "12" in first_message


class TestRunShard:
    """
    Tests for the RunShard RPC.
    """

    def test_run_shard_model_not_loaded(self):
        """
        RunShard for an unloaded model sets NOT_FOUND.
        """
        from inference.service import InferenceServiceServicer

        servicer = InferenceServiceServicer()
        context = MagicMock()

        response = servicer.RunShard(make_run_request(), context)

        assert response.success is False
        context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)

    def test_run_shard_generate_mode(self):
        """
        RunShard in generate mode calls model.generate and returns output tensor.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle()
        output_ids = torch.tensor([[101, 2023, 3045, 999, 888]], dtype=torch.int64)
        mock_handle.model.generate.return_value = output_ids

        servicer = InferenceServiceServicer()
        servicer._models[mock_handle.name] = mock_handle
        context = MagicMock()

        response = servicer.RunShard(make_run_request(generate_mode=True), context)

        assert response.success is True
        assert response.latency_ms > 0
        assert len(response.output_tensor) > 0

    def test_run_shard_forward_pass(self):
        """
        RunShard in forward-pass mode calls the model directly and uses
        its return value as the output tensor directly - MambaShardModule
        (what every handle wraps now) returns raw logits from forward(),
        not a HF-style output object with a .logits attribute.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle()
        mock_handle.model.return_value = torch.randn(1, 3, 50280)

        servicer = InferenceServiceServicer()
        servicer._models[mock_handle.name] = mock_handle
        context = MagicMock()

        response = servicer.RunShard(make_run_request(generate_mode=False), context)

        assert response.success is True
        assert len(response.output_tensor) > 0

    def test_run_shard_records_latency(self):
        """
        RunShard records a positive latency in the response.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle()
        mock_handle.model.generate.return_value = torch.tensor(
            [[1, 2, 3]], dtype=torch.int64
        )

        servicer = InferenceServiceServicer()
        servicer._models[mock_handle.name] = mock_handle
        context = MagicMock()

        response = servicer.RunShard(make_run_request(), context)

        assert response.latency_ms > 0

    @patch("inference.service.WorkerClient")
    def test_run_shard_pipeline_forwards_to_next_worker(self, mock_worker_cls):
        """
        In pipeline mode with a next_worker_address, RunShard fires
        WorkerClient.run_shard and returns an acknowledgement only.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle(next_worker_address="192.168.1.11:50052")
        mock_handle.model.return_value = torch.randn(1, 3, 768)

        mock_worker_instance = mock_worker_cls.return_value.__enter__.return_value

        servicer = InferenceServiceServicer()
        servicer._models[mock_handle.name] = mock_handle
        context = MagicMock()

        response = servicer.RunShard(make_pipeline_run_request(), context)

        assert response.success is True
        mock_worker_cls.assert_called_once_with("192.168.1.11:50052")
        mock_worker_instance.run_shard.assert_called_once()

    @patch("inference.service.PipelineCallbackClient")
    def test_run_shard_last_shard_delivers_result(self, mock_callback_cls):
        """
        In pipeline mode with an empty next_worker_address, RunShard calls
        PipelineCallbackClient.deliver_result on the orchestrator.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle(next_worker_address="")
        mock_handle.model.return_value = torch.randn(1, 3, 50280)

        mock_callback_instance = mock_callback_cls.return_value.__enter__.return_value

        servicer = InferenceServiceServicer()
        servicer._models[mock_handle.name] = mock_handle
        context = MagicMock()

        response = servicer.RunShard(make_pipeline_run_request(), context)

        assert response.success is True
        mock_callback_cls.assert_called_once_with("localhost:50060")
        mock_callback_instance.deliver_result.assert_called_once()


def make_real_single_shard_handle(model_name="cache-test-model"):
    """
    Build a ModelHandle wrapping a real (tiny, random-weight) single
    Mamba shard covering all layers - for testing RunShard's actual
    cache lifecycle logic, not just its plumbing around a mock.

    Parameters
    ----------
    model_name : str
        Name to give the handle.

    Returns
    -------
    ModelHandle
        A handle with a real MambaShardModule as its model, cache=None.
    """
    config = MambaConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=2, state_size=4
    )
    model = MambaForCausalLM(config)
    model.eval()
    shard = MambaShardModule.from_model(
        model, layer_start=0, layer_end=2, is_first=True, is_last=True
    )
    return ModelHandle(
        name=model_name,
        model=shard,
        tokenizer=None,
        manifest=None,
        memory_mb=1,
        loaded_at=0.0,
        layer_start=0,
        layer_end=2,
        is_first_shard=True,
        is_last_shard=True,
        next_worker_address="",
    )


class TestRunShardCache:
    """
    Tests for recurrent-state caching in RunShard's pipeline-mode path.
    Uses a real (tiny) shard directly injected into the servicer's
    model dict, rather than going through LoadShard, to exercise
    actual cache behavior rather than mocked plumbing.
    """

    @patch("inference.service.PipelineCallbackClient")
    def test_reset_cache_true_builds_a_fresh_cache(self, mock_callback_cls):
        """
        A reset_cache=True call builds a new cache on the handle.
        """
        from inference.service import InferenceServiceServicer

        servicer = InferenceServiceServicer()
        handle = make_real_single_shard_handle()
        servicer._models[handle.name] = handle
        assert handle.cache is None

        request = make_pipeline_run_request(
            model_name=handle.name,
            input_tensor=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        )
        request.reset_cache = True
        response = servicer.RunShard(request, MagicMock())

        assert response.success is True
        assert handle.cache is not None

    @patch("inference.service.PipelineCallbackClient")
    def test_reset_cache_false_reuses_existing_cache(self, mock_callback_cls):
        """
        A reset_cache=False call reuses the same cache object instead
        of rebuilding it from scratch.
        """
        from inference.service import InferenceServiceServicer

        servicer = InferenceServiceServicer()
        handle = make_real_single_shard_handle()
        servicer._models[handle.name] = handle

        first = make_pipeline_run_request(
            model_name=handle.name,
            input_tensor=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        )
        first.reset_cache = True
        servicer.RunShard(first, MagicMock())
        cache_after_first = handle.cache
        assert cache_after_first is not None

        second = make_pipeline_run_request(
            model_name=handle.name,
            input_tensor=torch.tensor([[5]], dtype=torch.int64),
        )
        second.reset_cache = False
        servicer.RunShard(second, MagicMock())

        assert handle.cache is cache_after_first

    @patch("inference.service.PipelineCallbackClient")
    def test_reset_cache_true_again_rebuilds_cache(self, mock_callback_cls):
        """
        A second reset_cache=True call (e.g. a fresh, unrelated
        generation against the same resident model) discards the old
        cache and builds a new one, so no state leaks between
        generations.
        """
        from inference.service import InferenceServiceServicer

        servicer = InferenceServiceServicer()
        handle = make_real_single_shard_handle()
        servicer._models[handle.name] = handle

        first = make_pipeline_run_request(
            model_name=handle.name,
            input_tensor=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        )
        first.reset_cache = True
        servicer.RunShard(first, MagicMock())
        cache_after_first = handle.cache

        second = make_pipeline_run_request(
            model_name=handle.name,
            input_tensor=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        )
        second.reset_cache = True
        servicer.RunShard(second, MagicMock())

        assert handle.cache is not cache_after_first

    def test_cached_decode_matches_full_recompute(self):
        """
        Prefill + a cached decode step, driven through the real RunShard
        RPC handler, produce the same final-position logits as a single
        RunShard call recomputing the whole sequence with no cache -
        the actual point of this feature, verified at the service layer
        rather than just the shard layer.
        """
        from inference.service import InferenceServiceServicer

        torch.manual_seed(0)
        vocab = 32
        prompt = torch.randint(0, vocab, (1, 3), dtype=torch.int64)
        next_token = torch.randint(0, vocab, (1, 1), dtype=torch.int64)
        full_sequence = torch.cat([prompt, next_token], dim=1)

        def run_and_capture(handle, input_tensor, reset_cache):
            servicer = InferenceServiceServicer()
            servicer._models[handle.name] = handle
            request = make_pipeline_run_request(
                model_name=handle.name, input_tensor=input_tensor
            )
            request.reset_cache = reset_cache
            with patch("inference.service.PipelineCallbackClient") as mock_cls:
                delivered = []
                mock_cls.return_value.__enter__.return_value.deliver_result.side_effect = (
                    delivered.append
                )
                servicer.RunShard(request, MagicMock())
            return deserialize_tensor(
                delivered[0].output_tensor,
                list(delivered[0].output_shape),
                delivered[0].output_dtype,
            )

        torch.manual_seed(1)
        reference_handle = make_real_single_shard_handle()
        reference_output = run_and_capture(
            reference_handle, full_sequence, reset_cache=True
        )

        torch.manual_seed(1)
        cached_handle = make_real_single_shard_handle()
        run_and_capture(cached_handle, prompt, reset_cache=True)
        cached_output = run_and_capture(cached_handle, next_token, reset_cache=False)

        assert torch.allclose(
            reference_output[0, -1, :], cached_output[0, -1, :], atol=1e-4
        )


class TestUnloadShard:
    """
    Tests for the UnloadShard RPC.
    """

    @patch("inference.service.unload_model")
    def test_unload_shard_success(self, mock_unload):
        """
        UnloadShard removes the model and returns success.
        """
        from inference.service import InferenceServiceServicer

        mock_unload.return_value = 260

        servicer = InferenceServiceServicer()
        servicer._models["mamba-130m"] = make_mock_handle()
        context = MagicMock()

        response = servicer.UnloadShard(
            inference_pb2.UnloadShardRequest(model_name="mamba-130m"), context
        )

        assert response.success is True
        assert response.memory_freed_mb == 260
        assert servicer._models.get("mamba-130m") is None

    def test_unload_shard_not_loaded(self):
        """
        UnloadShard for an unloaded model returns success=False.
        """
        from inference.service import InferenceServiceServicer

        servicer = InferenceServiceServicer()
        context = MagicMock()

        response = servicer.UnloadShard(
            inference_pb2.UnloadShardRequest(model_name="nonexistent"), context
        )

        assert response.success is False


class TestFormatError:
    """
    Tests for _format_error: compacts known exception shapes to one line.
    """

    def test_grpc_error_compacted(self):
        """
        A grpc.RpcError with code()/details() is rendered as "CODE: details".
        """
        from inference.service import _format_error

        err = grpc.RpcError()
        err.code = lambda: grpc.StatusCode.UNAVAILABLE
        err.details = lambda: "Connection refused"

        assert _format_error(err) == "StatusCode.UNAVAILABLE: Connection refused"

    def test_http_error_with_response_compacted(self):
        """
        An httpx-based HTTP error (e.g. huggingface_hub's HfHubHTTPError,
        raised when a checkpoint can't be resolved) is rendered as a
        single "HTTP <code> <reason> for <url>" line, not str(e)'s
        multi-paragraph dump (status line, blank line, explanation,
        "please make sure...", a docs link).
        """
        from inference.service import _format_error

        response = MagicMock()
        response.status_code = 404
        response.reason_phrase = "Not Found"
        response.url = "https://huggingface.co/checkpoints/dummy-mamba-tiny/resolve/main/model.safetensors"

        err = httpx.HTTPError(
            "404 Client Error. (Request ID: ...)\n\nRepository Not Found for url: "
            "...\nPlease make sure you specified the correct `repo_id` and "
            "`repo_type`.\nFor more details, see https://huggingface.co/docs/..."
        )
        err.response = response

        formatted = _format_error(err)

        assert formatted == (
            "HTTP 404 Not Found for https://huggingface.co/checkpoints/"
            "dummy-mamba-tiny/resolve/main/model.safetensors"
        )
        assert "\n" not in formatted
        assert "please make sure" not in formatted.lower()

    def test_plain_exception_falls_back_to_str(self):
        """
        An exception type with no special handling falls back to str(e).
        """
        from inference.service import _format_error

        assert _format_error(ValueError("bad arch 's4'")) == "bad arch 's4'"
