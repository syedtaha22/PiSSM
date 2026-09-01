"""
Unit tests for the model loader.

Tests the loader interface and error handling using mocked
transformers calls. No model download needed.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from inference.manifest import ModelManifest


def make_manifest(arch="mamba"):
    """
    Create a ModelManifest with sensible defaults.

    Parameters
    ----------
    arch : str
        Architecture for the manifest.

    Returns
    -------
    ModelManifest
        A valid manifest instance.
    """
    return ModelManifest(
        name="test-model",
        arch=arch,
        checkpoint="test/checkpoint",
        layers=24,
        hidden_dim=768,
        state_dim=16,
        input_type="text",
        tokenizer="test/tokenizer",
    )


class TestArchToModelClass:
    """
    Tests for the _ARCH_TO_MODEL_CLASS registry.
    """

    def test_falcon_mamba_registered(self):
        """
        "falcon-mamba" resolves to transformers' FalconMambaForCausalLM.
        """
        from transformers import FalconMambaForCausalLM

        from inference.loader import _ARCH_TO_MODEL_CLASS

        assert _ARCH_TO_MODEL_CLASS["falcon-mamba"] is FalconMambaForCausalLM


class TestLoadModel:
    """
    Tests for the load_model function.
    """

    def test_unsupported_arch_raises(self):
        """
        Loading a model with an unsupported architecture raises
        NotImplementedError.
        """
        from inference.loader import load_model

        manifest = make_manifest(arch="s4")
        with pytest.raises(NotImplementedError, match="s4"):
            load_model(manifest)

    def make_mock_model(self):
        """
        Create a mock model with empty parameters/buffers iterables.

        Returns
        -------
        MagicMock
            A mock model safe to pass through _weight_bytes_mb.
        """
        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        return mock_model

    @patch("inference.loader.AutoTokenizer")
    def test_load_returns_model_handle(self, mock_tokenizer_cls):
        """
        Loading a supported model returns a ModelHandle with all fields set.
        """
        from inference.loader import _ARCH_TO_MODEL_CLASS, load_model

        mock_model = self.make_mock_model()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        manifest = make_manifest()
        with patch.dict(_ARCH_TO_MODEL_CLASS, {"mamba": mock_model_cls}):
            handle = load_model(manifest)

        assert handle.name == "test-model"
        assert handle.model is mock_model
        assert handle.tokenizer is mock_tokenizer
        assert handle.manifest is manifest
        assert handle.loaded_at > 0

    @patch("inference.loader.AutoTokenizer")
    def test_load_sets_eval_mode(self, mock_tokenizer_cls):
        """
        The loaded model is set to eval mode.
        """
        from inference.loader import _ARCH_TO_MODEL_CLASS, load_model

        mock_model = self.make_mock_model()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch.dict(_ARCH_TO_MODEL_CLASS, {"mamba": mock_model_cls}):
            load_model(make_manifest())

        mock_model.eval.assert_called_once()

    @patch("inference.loader.AutoTokenizer")
    def test_load_sets_cpu(self, mock_tokenizer_cls):
        """
        The loaded model is moved to CPU.
        """
        from inference.loader import _ARCH_TO_MODEL_CLASS, load_model

        mock_model = self.make_mock_model()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch.dict(_ARCH_TO_MODEL_CLASS, {"mamba": mock_model_cls}):
            load_model(make_manifest())

        mock_model.to.assert_called_once_with("cpu")

    @patch("inference.loader.AutoTokenizer")
    def test_load_reports_weight_byte_size(self, mock_tokenizer_cls):
        """
        memory_mb reflects the sum of parameter and buffer tensor bytes,
        not a live RSS reading.
        """
        from inference.loader import _ARCH_TO_MODEL_CLASS, load_model

        mock_model = MagicMock()
        # 1000 float32 elements = 4000 bytes = 0 MB (integer division),
        # plus a second parameter to push past a whole MB boundary.
        param = torch.nn.Parameter(torch.zeros(300_000))  # 1,200,000 bytes
        buffer = torch.zeros(50_000)  # 200,000 bytes
        mock_model.parameters.return_value = [param]
        mock_model.buffers.return_value = [buffer]
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch.dict(_ARCH_TO_MODEL_CLASS, {"mamba": mock_model_cls}):
            handle = load_model(make_manifest())

        expected_mb = (1_200_000 + 200_000) // (1024 * 1024)
        assert handle.memory_mb == expected_mb


class TestUnloadModel:
    """
    Tests for the unload_model function.
    """

    def test_unload_clears_references(self):
        """
        After unloading, model and tokenizer are set to None.
        """
        from inference.loader import ModelHandle, unload_model

        handle = ModelHandle(
            name="test",
            model=MagicMock(),
            tokenizer=MagicMock(),
            manifest=make_manifest(),
            memory_mb=100,
            loaded_at=0.0,
        )

        unload_model(handle)

        assert handle.model is None
        assert handle.tokenizer is None

    def test_unload_reports_the_handles_own_memory_mb(self):
        """
        The freed amount is the handle's already-known memory_mb, not
        a live RSS measurement - freeing Python references doesn't
        guarantee the allocator returns pages to the OS immediately,
        so an RSS delta can't be trusted to reflect what was freed.
        """
        from inference.loader import ModelHandle, unload_model

        handle = ModelHandle(
            name="test",
            model=MagicMock(),
            tokenizer=MagicMock(),
            manifest=make_manifest(),
            memory_mb=260,
            loaded_at=0.0,
        )

        assert unload_model(handle) == 260
