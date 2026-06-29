"""
Tests for the InferenceServiceServicer gRPC handler.

Validates that the servicer correctly bridges gRPC requests to the
model loader. Tests call servicer methods directly with mock context
objects and a mocked loader, avoiding any model downloads or network I/O.
"""

from unittest.mock import MagicMock, patch

import grpc
import torch

from proto.generated import inference_pb2
from inference.tensor_utils import serialize_tensor


def make_load_request(
    model_name="mamba-130m",
    checkpoint="state-spaces/mamba-130m-hf",
    tokenizer="EleutherAI/gpt-neox-20b",
    arch="mamba",
    layer_start=0,
    layer_end=0,
):
    """
    Build a LoadShardRequest with sensible defaults.

    Parameters
    ----------
    model_name : str
        Model name.
    checkpoint : str
        HuggingFace model ID.
    tokenizer : str
        HuggingFace tokenizer ID.
    arch : str
        Architecture string.
    layer_start : int
        First layer index.
    layer_end : int
        Last layer index.

    Returns
    -------
    inference_pb2.LoadShardRequest
        A populated load request.
    """
    return inference_pb2.LoadShardRequest(
        model_name=model_name,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        arch=arch,
        layer_start=layer_start,
        layer_end=layer_end,
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


def make_mock_handle(name="mamba-130m"):
    """
    Create a mock ModelHandle.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    MagicMock
        A mock handle with expected attributes.
    """
    handle = MagicMock()
    handle.name = name
    handle.memory_mb = 260
    handle.manifest = MagicMock()
    handle.manifest.layers = 24
    return handle


class TestLoadShard:
    """
    Tests for the LoadShard RPC.
    """

    @patch("inference.service.load_model")
    def test_load_shard_success(self, mock_load):
        """
        LoadShard with valid parameters returns success=True.
        """
        from inference.service import InferenceServiceServicer

        mock_load.return_value = make_mock_handle()
        servicer = InferenceServiceServicer()
        context = MagicMock()

        response = servicer.LoadShard(make_load_request(), context)

        assert response.success is True
        assert response.error_message == ""
        assert response.memory_used_mb == 260
        assert response.layers_loaded == 24

    @patch("inference.service.load_model")
    def test_load_shard_stores_model(self, mock_load):
        """
        After LoadShard, the model is stored and retrievable.
        """
        from inference.service import InferenceServiceServicer

        mock_load.return_value = make_mock_handle()
        servicer = InferenceServiceServicer()
        context = MagicMock()

        servicer.LoadShard(make_load_request(), context)

        assert servicer._models.get("mamba-130m") is not None

    @patch("inference.service.load_model")
    def test_load_shard_duplicate_rejects(self, mock_load):
        """
        Loading the same model name twice returns success=False.
        """
        from inference.service import InferenceServiceServicer

        mock_load.return_value = make_mock_handle()
        servicer = InferenceServiceServicer()
        context = MagicMock()

        servicer.LoadShard(make_load_request(), context)
        response = servicer.LoadShard(make_load_request(), context)

        assert response.success is False
        assert "already loaded" in response.error_message

    @patch("inference.service.load_model")
    def test_load_shard_failure(self, mock_load):
        """
        LoadShard returns success=False when the loader raises.
        """
        from inference.service import InferenceServiceServicer

        mock_load.side_effect = NotImplementedError("arch 's4' not supported")
        servicer = InferenceServiceServicer()
        context = MagicMock()

        response = servicer.LoadShard(make_load_request(arch="s4"), context)

        assert response.success is False
        assert "s4" in response.error_message


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

    @patch("inference.service.load_model")
    def test_run_shard_generate_mode(self, mock_load):
        """
        RunShard in generate mode calls model.generate and returns output tensor.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle()
        output_ids = torch.tensor([[101, 2023, 3045, 999, 888]], dtype=torch.int64)
        mock_handle.model.generate.return_value = output_ids
        mock_load.return_value = mock_handle

        servicer = InferenceServiceServicer()
        context = MagicMock()

        servicer.LoadShard(make_load_request(), context)
        response = servicer.RunShard(make_run_request(generate_mode=True), context)

        assert response.success is True
        assert response.latency_ms > 0
        assert len(response.output_tensor) > 0

    @patch("inference.service.load_model")
    def test_run_shard_forward_pass(self, mock_load):
        """
        RunShard in forward-pass mode calls the model directly.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 50280)
        mock_handle.model.return_value = mock_output
        mock_load.return_value = mock_handle

        servicer = InferenceServiceServicer()
        context = MagicMock()

        servicer.LoadShard(make_load_request(), context)
        response = servicer.RunShard(make_run_request(generate_mode=False), context)

        assert response.success is True
        assert len(response.output_tensor) > 0

    @patch("inference.service.load_model")
    def test_run_shard_records_latency(self, mock_load):
        """
        RunShard records a positive latency in the response.
        """
        from inference.service import InferenceServiceServicer

        mock_handle = make_mock_handle()
        mock_handle.model.generate.return_value = torch.tensor(
            [[1, 2, 3]], dtype=torch.int64
        )
        mock_load.return_value = mock_handle

        servicer = InferenceServiceServicer()
        context = MagicMock()

        servicer.LoadShard(make_load_request(), context)
        response = servicer.RunShard(make_run_request(), context)

        assert response.latency_ms > 0


class TestUnloadShard:
    """
    Tests for the UnloadShard RPC.
    """

    @patch("inference.service.unload_model")
    @patch("inference.service.load_model")
    def test_unload_shard_success(self, mock_load, mock_unload):
        """
        UnloadShard removes the model and returns success.
        """
        from inference.service import InferenceServiceServicer

        mock_load.return_value = make_mock_handle()
        mock_unload.return_value = 260

        servicer = InferenceServiceServicer()
        context = MagicMock()

        servicer.LoadShard(make_load_request(), context)
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
