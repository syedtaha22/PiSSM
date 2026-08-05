"""
Tests for MambaShardModule.

Validates the shard module's forward pass shape for each pipeline
position (first, middle, last, single) using lightweight mock sub-modules
so no real model weights are required.
"""

import io

import pytest
import torch
import torch.nn as nn
from transformers import MambaConfig, MambaForCausalLM

from inference.shard import MambaShardModule


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

    def test_from_bytes_retains_full_model_config_for_sub_range_shard(self):
        """
        from_bytes() retains the full model's config (not a config
        sized to just this shard's own layer range) - this is what
        guards the cache-sizing landmine: Cache.layers[layer_idx] is
        indexed by the model's original, global layer_idx.
        """
        config = MambaConfig(
            vocab_size=32, hidden_size=8, num_hidden_layers=4, state_size=4
        )
        model = MambaForCausalLM(config)
        source_shard = MambaShardModule.from_model(
            model, layer_start=2, layer_end=4, is_first=False, is_last=True
        )
        buf = io.BytesIO()
        torch.save(source_shard.state_dict(), buf)

        shard = MambaShardModule.from_bytes(
            weights_bytes=buf.getvalue(),
            config_json_bytes=config.to_json_string().encode(),
            layer_start=2,
            layer_end=4,
            is_first=False,
            is_last=True,
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
