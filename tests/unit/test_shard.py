"""
Tests for MambaShardModule.

Validates the shard module's forward pass shape for each pipeline
position (first, middle, last, single) using lightweight mock sub-modules
so no real model weights are required.
"""

import pytest
import torch
from torch import nn
from transformers import AutoConfig, MambaConfig, MambaForCausalLM

from inference.shard import MambaShardModule
from inference.weights import read_checkpoint_metadata

DUMMY_CHECKPOINT = "checkpoints/dummy-mamba-tiny"


class IdentityLayer(nn.Module):
    """
    Stand-in for a MambaBlock: passes hidden states through unchanged,
    accepting (and ignoring) cache_params like the real layer's
    signature does.
    """

    def forward(self, x, cache_params=None):
        return x


class TestMambaShardModuleForward:
    """
    Tests for MambaShardModule forward pass output shapes.
    """

    def test_first_shard_forward_shape(self):
        """
        First shard: token IDs in, hidden states out.

        Embedding converts (batch, seq) int IDs to (batch, seq, d_model)
        float hidden states. The layer passes them through unchanged.
        """
        batch, seq, d_model, vocab = 1, 5, 8, 100

        shard = MambaShardModule(
            layers=[IdentityLayer()],
            is_first=True,
            is_last=False,
            embeddings=nn.Embedding(vocab, d_model),
        )

        input_ids = torch.randint(0, vocab, (batch, seq))
        output = shard(input_ids)

        assert output.shape == (batch, seq, d_model)

    def test_middle_shard_forward_shape(self):
        """
        Middle shard: hidden states in, hidden states out.

        No embedding, norm, or lm_head applied; layer passes state through.
        """
        batch, seq, d_model = 1, 5, 8

        shard = MambaShardModule(
            layers=[IdentityLayer()],
            is_first=False,
            is_last=False,
        )

        hidden = torch.randn(batch, seq, d_model)
        output = shard(hidden)

        assert output.shape == (batch, seq, d_model)

    def test_last_shard_forward_shape(self):
        """
        Last shard: hidden states in, logits out.

        After the layer, norm_f and lm_head project to vocabulary size.
        """
        batch, seq, d_model, vocab = 1, 5, 8, 100

        shard = MambaShardModule(
            layers=[IdentityLayer()],
            is_first=False,
            is_last=True,
            norm_f=nn.LayerNorm(d_model),
            lm_head=nn.Linear(d_model, vocab, bias=False),
        )

        hidden = torch.randn(batch, seq, d_model)
        output = shard(hidden)

        assert output.shape == (batch, seq, vocab)

    def test_single_shard_forward_shape(self):
        """
        Single shard (is_first and is_last): token IDs in, logits out.

        Embedding, layer, norm, and lm_head all applied in sequence.
        """
        batch, seq, d_model, vocab = 1, 5, 8, 100

        shard = MambaShardModule(
            layers=[IdentityLayer()],
            is_first=True,
            is_last=True,
            embeddings=nn.Embedding(vocab, d_model),
            norm_f=nn.LayerNorm(d_model),
            lm_head=nn.Linear(d_model, vocab, bias=False),
        )

        input_ids = torch.randint(0, vocab, (batch, seq))
        output = shard(input_ids)

        assert output.shape == (batch, seq, vocab)

    def test_multiple_layers_applied_in_order(self):
        """
        Each layer in the ModuleList is applied sequentially.

        Two layers that each scale by a factor of 2 produce an output
        scaled by 4 relative to the input hidden states.
        """

        class ScaleBy2(nn.Module):
            def forward(self, x, cache_params=None):
                return x * 2

        batch, seq, d_model = 1, 3, 4

        shard = MambaShardModule(
            layers=[ScaleBy2(), ScaleBy2()],
            is_first=False,
            is_last=False,
        )

        hidden = torch.ones(batch, seq, d_model)
        output = shard(hidden)

        assert torch.allclose(output, torch.full_like(output, 4.0))


class TestMambaShardModuleCache:
    """
    Tests for cache_params threading and full-config retention, needed
    for recurrent-state caching across the pipeline.
    """

    def test_forward_threads_cache_params_into_each_layer(self):
        """
        forward()'s cache_params argument is passed through to every
        layer's forward call, not silently dropped.
        """

        class RecordingLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.received = "never called"

            def forward(self, x, cache_params=None):
                self.received = cache_params
                return x

        layer = RecordingLayer()
        shard = MambaShardModule(layers=[layer], is_first=False, is_last=False)
        sentinel = object()

        shard(torch.randn(1, 2, 4), cache_params=sentinel)

        assert layer.received is sentinel

    def test_from_model_retains_full_model_config(self):
        """
        from_model() keeps a reference to the *full* model's config,
        even for a shard covering only a sub-range of its layers.
        """
        config = MambaConfig(
            vocab_size=32, hidden_size=8, num_hidden_layers=4, state_size=4
        )
        model = MambaForCausalLM(config)

        shard = MambaShardModule.from_model(
            model, layer_start=2, layer_end=4, is_first=False, is_last=True
        )

        assert shard.config is model.config
        assert shard.config.num_hidden_layers == 4

    def test_from_tensor_locations_retains_full_model_config_for_sub_range_shard(self):
        """
        from_tensor_locations() retains the full model's config (not a
        config sized to just this shard's own layer range) - this is
        what guards the cache-sizing landmine: Cache.layers[layer_idx]
        is indexed by the model's original, global layer_idx.
        """
        config = AutoConfig.from_pretrained(DUMMY_CHECKPOINT)
        weight_map = read_checkpoint_metadata(DUMMY_CHECKPOINT).weight_map
        locations = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=2, layer_end=4, is_first=False, is_last=True
        )

        shard = MambaShardModule.from_tensor_locations(
            config,
            layer_start=2,
            layer_end=4,
            is_first=False,
            is_last=True,
            checkpoint=DUMMY_CHECKPOINT,
            tensor_locations=locations,
        )

        assert shard.config.num_hidden_layers == 4
        assert len(shard.layers) == 2

    def test_new_cache_sized_from_full_config_not_shard_layer_count(self):
        """
        new_cache() must size the cache from the full model's layer
        count (self.config.num_hidden_layers), never len(self.layers) -
        the exact landmine this feature has to avoid reintroducing.
        """
        config = MambaConfig(
            vocab_size=32, hidden_size=8, num_hidden_layers=4, state_size=4
        )
        model = MambaForCausalLM(config)
        shard = MambaShardModule.from_model(
            model, layer_start=2, layer_end=4, is_first=False, is_last=True
        )

        cache = shard.new_cache()

        assert len(cache.layers) == 4
        assert len(cache.layers) != len(shard.layers)


class TestMambaShardModuleValidation:
    """
    Tests for MambaShardModule constructor validation.
    """

    def test_requires_embeddings_when_is_first(self):
        """
        Raises ValueError when is_first=True but embeddings is None.
        """
        with pytest.raises(ValueError, match="embeddings"):
            MambaShardModule(layers=[], is_first=True, is_last=False)

    def test_requires_norm_and_lm_head_when_is_last(self):
        """
        Raises ValueError when is_last=True but norm_f or lm_head is None.
        """
        with pytest.raises(ValueError, match="norm_f"):
            MambaShardModule(layers=[], is_first=False, is_last=True)


class TestTensorLocationsForRange:
    """
    Tests for MambaShardModule.tensor_locations_for_range: pure
    tensor-name-to-shard resolution from a checkpoint's weight_map, no
    torch or network I/O.
    """

    def test_first_shard_includes_embeddings(self):
        """
        A first shard's locations include the embedding tensor; a
        middle/last shard's do not.
        """
        weight_map = {
            "backbone.embeddings.weight": "model.safetensors",
            "backbone.layers.0.mixer.A_log": "model.safetensors",
            "backbone.norm_f.weight": "model.safetensors",
        }

        first_locs = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=0, layer_end=1, is_first=True, is_last=False
        )
        middle_locs = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=0, layer_end=1, is_first=False, is_last=False
        )

        assert any(loc.dest_name == "embeddings.weight" for loc in first_locs)
        assert not any(loc.dest_name == "embeddings.weight" for loc in middle_locs)

    def test_only_layers_in_range_included_with_renumbered_dest_name(self):
        """
        Only tensors for layers in [layer_start, layer_end) are included,
        and dest_name is renumbered to the shard's own local layer index
        (layer_idx - layer_start), not the checkpoint's global index.
        """
        weight_map = {
            "backbone.layers.2.mixer.A_log": "model.safetensors",
            "backbone.layers.3.mixer.A_log": "model.safetensors",
            "backbone.layers.4.mixer.A_log": "model.safetensors",
        }

        locs = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=2, layer_end=4, is_first=False, is_last=False
        )

        dest_names = {loc.dest_name for loc in locs}
        assert dest_names == {"layers.0.mixer.A_log", "layers.1.mixer.A_log"}

    def test_shard_spanning_file_boundary_resolves_both_files(self):
        """
        A shard's tensor_locations can reference more than one distinct
        source_file when its layer range straddles a checkpoint's own
        file-sharding boundary - HuggingFace splits multi-file
        checkpoints (e.g. mamba-1.4b-hf, mamba-2.8b-hf) by byte size,
        not by layer count, so a file boundary can fall mid-layer-range.
        Synthetic weight_map here, deliberately not aligned with the
        shard boundary below, so this needs no real checkpoint or
        network access to verify.
        """
        weight_map = {
            "backbone.layers.0.mixer.A_log": "model-00001-of-00002.safetensors",
            "backbone.layers.1.mixer.A_log": "model-00001-of-00002.safetensors",
            "backbone.layers.2.mixer.A_log": "model-00002-of-00002.safetensors",
            "backbone.layers.3.mixer.A_log": "model-00002-of-00002.safetensors",
        }

        locs = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=1, layer_end=3, is_first=False, is_last=False
        )

        source_files = {loc.source_file for loc in locs}
        assert source_files == {
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        }

    def test_last_shard_includes_norm_f_and_untied_lm_head(self):
        """
        A last shard's locations include norm_f and lm_head when
        lm_head.weight has its own entry in the weight map (untied).
        """
        weight_map = {
            "backbone.layers.0.mixer.A_log": "model.safetensors",
            "backbone.norm_f.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
        }

        locs = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=0, layer_end=1, is_first=False, is_last=True
        )

        by_dest = {loc.dest_name: loc for loc in locs}
        assert "norm_f.weight" in by_dest
        assert by_dest["lm_head.weight"].source_name == "lm_head.weight"

    def test_last_shard_tied_lm_head_resolves_to_embeddings(self):
        """
        When lm_head.weight is absent from the weight map (tied to the
        input embedding, mamba's default), the last shard's lm_head
        location sources from backbone.embeddings.weight instead.
        """
        weight_map = {
            "backbone.embeddings.weight": "model.safetensors",
            "backbone.layers.0.mixer.A_log": "model.safetensors",
            "backbone.norm_f.weight": "model.safetensors",
        }

        locs = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=0, layer_end=1, is_first=False, is_last=True
        )

        by_dest = {loc.dest_name: loc for loc in locs}
        assert by_dest["lm_head.weight"].source_name == "backbone.embeddings.weight"
        assert by_dest["lm_head.weight"].source_file == "model.safetensors"


class TestFromTensorLocations:
    """
    Tests for MambaShardModule.from_tensor_locations against the real
    local dummy-mamba-tiny checkpoint - fast, no network, but a real
    safetensors file with the same tied-weight shape as production
    checkpoints.
    """

    def _dummy_config(self):
        return AutoConfig.from_pretrained(DUMMY_CHECKPOINT)

    def test_matches_directly_loaded_reference_for_first_shard(self):
        """
        A first shard built via from_tensor_locations has the same
        embedding and first-layer weights as slicing a directly
        from_pretrained-loaded reference model.
        """
        config = self._dummy_config()
        weight_map = read_checkpoint_metadata(DUMMY_CHECKPOINT).weight_map
        reference = MambaForCausalLM.from_pretrained(DUMMY_CHECKPOINT)
        reference_shard = MambaShardModule.from_model(
            reference, layer_start=0, layer_end=2, is_first=True, is_last=False
        )

        locations = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=0, layer_end=2, is_first=True, is_last=False
        )
        shard = MambaShardModule.from_tensor_locations(
            config,
            layer_start=0,
            layer_end=2,
            is_first=True,
            is_last=False,
            checkpoint=DUMMY_CHECKPOINT,
            tensor_locations=locations,
        )

        reference_sd = reference_shard.state_dict()
        shard_sd = shard.state_dict()
        assert set(shard_sd.keys()) == set(reference_sd.keys())
        for key in shard_sd:
            assert torch.equal(shard_sd[key], reference_sd[key])

    def test_matches_directly_loaded_reference_for_tied_last_shard(self):
        """
        A last shard (not also first) built via from_tensor_locations
        gets a correct lm_head weight even though it's tied to the
        embedding and has no storage of its own in the checkpoint.
        """
        config = self._dummy_config()
        weight_map = read_checkpoint_metadata(DUMMY_CHECKPOINT).weight_map
        reference = MambaForCausalLM.from_pretrained(DUMMY_CHECKPOINT)
        reference_shard = MambaShardModule.from_model(
            reference, layer_start=2, layer_end=4, is_first=False, is_last=True
        )

        locations = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=2, layer_end=4, is_first=False, is_last=True
        )
        shard = MambaShardModule.from_tensor_locations(
            config,
            layer_start=2,
            layer_end=4,
            is_first=False,
            is_last=True,
            checkpoint=DUMMY_CHECKPOINT,
            tensor_locations=locations,
        )

        assert torch.equal(shard.lm_head.weight, reference_shard.lm_head.weight)
        assert torch.equal(shard.norm_f.weight, reference_shard.norm_f.weight)

    def test_fetches_each_distinct_source_file_once(self):
        """
        from_tensor_locations fetches each distinct source_file exactly
        once even when multiple tensor locations share it - dummy-mamba-
        tiny is single-file, so every location shares the same file.
        """
        config = self._dummy_config()
        weight_map = read_checkpoint_metadata(DUMMY_CHECKPOINT).weight_map
        locations = MambaShardModule.tensor_locations_for_range(
            weight_map, layer_start=0, layer_end=2, is_first=True, is_last=False
        )
        assert len(locations) > 1

        from unittest.mock import patch

        with patch(
            "inference.shard.fetch_source_file",
            return_value=__import__("pathlib").Path(DUMMY_CHECKPOINT)
            / "model.safetensors",
        ) as mock_fetch:
            MambaShardModule.from_tensor_locations(
                config,
                layer_start=0,
                layer_end=2,
                is_first=True,
                is_last=False,
                checkpoint=DUMMY_CHECKPOINT,
                tensor_locations=locations,
            )

        assert mock_fetch.call_count == 1
