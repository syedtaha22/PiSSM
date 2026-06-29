"""
Unit tests for the model loader.

Tests the loader interface and error handling using mocked
transformers calls. No model download needed.
"""

from unittest.mock import MagicMock, patch

import pytest

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

    @patch("inference.loader.AutoTokenizer")
    def test_load_returns_model_handle(self, mock_tokenizer_cls):
        """
        Loading a supported model returns a ModelHandle with all fields set.
        """
        from inference.loader import _ARCH_TO_MODEL_CLASS, load_model

        mock_model = MagicMock()
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

        mock_model = MagicMock()
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

        mock_model = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch.dict(_ARCH_TO_MODEL_CLASS, {"mamba": mock_model_cls}):
            load_model(make_manifest())

        mock_model.to.assert_called_once_with("cpu")


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
