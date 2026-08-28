"""
Tests for ModelStore.

ModelStore reads a checkpoint's safetensors header/index and config
(never its weight bytes) and computes per-shard tensor-name assignments
by delegating to the registered architecture's shard class. Tests mock
inference.weights.read_checkpoint_metadata, AutoConfig, and the shard
class registry, so no torch model or network I/O is involved.
"""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import are_progress_bars_disabled

from orchestrator.model_store import ModelStore


def make_manifest(checkpoint="state-spaces/mamba-130m-hf"):
    """
    Return a minimal mock manifest exposing only what ModelStore reads.
    """
    manifest = MagicMock()
    manifest.checkpoint = checkpoint
    return manifest


class TestLoad:
    """
    Tests for ModelStore.load: reads checkpoint metadata and config only.
    """

    @patch("orchestrator.model_store.AutoConfig")
    @patch("orchestrator.model_store.read_checkpoint_metadata")
    def test_load_reads_metadata_and_config(self, mock_read_metadata, mock_auto_config):
        """
        load() calls read_checkpoint_metadata and AutoConfig.from_pretrained
        with the manifest's checkpoint - never a full model from_pretrained.
        """
        mock_read_metadata.return_value = MagicMock(weight_map={})
        mock_config = MagicMock()
        mock_config.to_json_string.return_value = "{}"
        mock_auto_config.from_pretrained.return_value = mock_config

        store = ModelStore()
        store.load(make_manifest("state-spaces/mamba-130m-hf"))

        mock_read_metadata.assert_called_once_with("state-spaces/mamba-130m-hf")
        mock_auto_config.from_pretrained.assert_called_once_with(
            "state-spaces/mamba-130m-hf"
        )


class TestExtractShard:
    """
    Tests for ModelStore.extract_shard: delegates tensor-name resolution
    to the registered shard class.
    """

    @patch("orchestrator.model_store._ARCH_TO_SHARD_CLASS")
    @patch("orchestrator.model_store.AutoConfig")
    @patch("orchestrator.model_store.read_checkpoint_metadata")
    def test_extract_shard_delegates_to_shard_class(
        self, mock_read_metadata, mock_auto_config, mock_registry
    ):
        """
        extract_shard() calls the registered shard class's
        tensor_locations_for_range with the weight_map and shard range,
        and returns (locations, config_json_bytes).
        """
        weight_map = {"backbone.embeddings.weight": "model.safetensors"}
        mock_read_metadata.return_value = MagicMock(weight_map=weight_map)
        mock_config = MagicMock()
        mock_config.to_json_string.return_value = '{"hidden_size": 8}'
        mock_auto_config.from_pretrained.return_value = mock_config

        mock_shard_cls = MagicMock()
        mock_shard_cls.tensor_locations_for_range.return_value = ["loc1", "loc2"]
        mock_registry.get.return_value = mock_shard_cls

        store = ModelStore()
        store.load(make_manifest())
        locations, config_json = store.extract_shard(
            "mamba", layer_start=0, layer_end=12, is_first=True, is_last=False
        )

        mock_shard_cls.tensor_locations_for_range.assert_called_once_with(
            weight_map, 0, 12, True, False
        )
        assert locations == ["loc1", "loc2"]
        assert config_json == b'{"hidden_size": 8}'

    def test_extract_shard_raises_before_load(self):
        """
        extract_shard() raises RuntimeError if called before load().
        """
        store = ModelStore()

        with pytest.raises(RuntimeError, match="load"):
            store.extract_shard("mamba", 0, 12, True, False)

    @patch("orchestrator.model_store.AutoConfig")
    @patch("orchestrator.model_store.read_checkpoint_metadata")
    def test_extract_shard_raises_for_unregistered_arch(
        self, mock_read_metadata, mock_auto_config
    ):
        """
        extract_shard() raises NotImplementedError for an arch with no
        registered shard class.
        """
        mock_read_metadata.return_value = MagicMock(weight_map={})
        mock_config = MagicMock()
        mock_config.to_json_string.return_value = "{}"
        mock_auto_config.from_pretrained.return_value = mock_config

        store = ModelStore()
        store.load(make_manifest())

        with pytest.raises(NotImplementedError, match="s4"):
            store.extract_shard("s4", 0, 12, True, False)


class TestUnload:
    """
    Tests for ModelStore.unload: clears cached metadata.
    """

    @patch("orchestrator.model_store.AutoConfig")
    @patch("orchestrator.model_store.read_checkpoint_metadata")
    def test_unload_clears_metadata(self, mock_read_metadata, mock_auto_config):
        """
        After unload(), extract_shard() raises again as if never loaded.
        """
        mock_read_metadata.return_value = MagicMock(weight_map={})
        mock_config = MagicMock()
        mock_config.to_json_string.return_value = "{}"
        mock_auto_config.from_pretrained.return_value = mock_config

        store = ModelStore()
        store.load(make_manifest())
        store.unload()

        with pytest.raises(RuntimeError, match="load"):
            store.extract_shard("mamba", 0, 12, True, False)


class TestProgressBars:
    """
    The orchestrator never downloads weights, so huggingface_hub's own
    progress bars (which would otherwise show for the small config/
    tokenizer fetches this process makes) are disabled process-wide by
    importing this module. This is safe because the worker process
    never imports orchestrator.model_store, so its own weight-download
    progress bar (inference/weights.py) is unaffected.
    """

    def test_importing_this_module_disables_progress_bars(self):
        """
        Progress bars are disabled as a side effect of import - by the
        time this test runs, the module (imported at the top of this
        test file) has already run that side effect.
        """
        assert are_progress_bars_disabled() is True
