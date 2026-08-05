"""
Regression tests for recurrent-state caching in the sharded Mamba pipeline.

Builds a tiny in-memory MambaForCausalLM (random weights, no checkpoint
on disk) so these stay fast, self-contained unit tests with no file I/O
dependency - the numerical values are meaningless, only internal
consistency between the cached and non-cached paths matters.
"""

import torch
from transformers import MambaConfig, MambaForCausalLM

from inference.shard import MambaShardModule


def _tiny_model() -> MambaForCausalLM:
    """
    Build a tiny random-weight MambaForCausalLM for fast, offline tests.
    """
    config = MambaConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=4,
        state_size=4,
        conv_kernel=2,
    )
    model = MambaForCausalLM(config)
    model.eval()
    return model


class TestNewCacheSizing:
    """
    Guards the config-sizing landmine: a shard's cache must always be
    sized from the full model's layer count, never the shard's own
    (smaller) slice, since MambaBlock instances retain their original
    global layer_idx after slicing and Cache.layers[layer_idx] indexes
    positionally - a shard-local cache size raises IndexError the first
    time a non-zero-based layer_idx is used.
    """

    def test_new_cache_has_one_slot_per_full_model_layer(self):
        """
        A shard owning a strict sub-range still gets a cache sized to
        the total model's layer count, not its own (smaller) range.
        """
        model = _tiny_model()
        total_layers = model.config.num_hidden_layers

        shard = MambaShardModule.from_model(
            model,
            layer_start=2,
            layer_end=total_layers,
            is_first=False,
            is_last=True,
        )

        cache = shard.new_cache()

        assert len(cache.layers) == total_layers
        assert len(shard.layers) == total_layers - 2


class TestCachedDecodeParity:
    """
    Verifies a cached incremental decode step produces the same output
    as fully recomputing the growing sequence from scratch - the
    optimization must not change what gets generated, only how fast.
    """

    def test_cached_decode_matches_full_recompute(self):
        """
        Prefill + one cached decode step matches feeding the whole
        sequence through with no cache at all.
        """
        model = _tiny_model()
        total_layers = model.config.num_hidden_layers
        split = total_layers // 2

        shard1 = MambaShardModule.from_model(
            model, layer_start=0, layer_end=split, is_first=True, is_last=False
        )
        shard2 = MambaShardModule.from_model(
            model,
            layer_start=split,
            layer_end=total_layers,
            is_first=False,
            is_last=True,
        )
        shard1.eval()
        shard2.eval()

        torch.manual_seed(0)
        prompt = torch.randint(0, model.config.vocab_size, (1, 3))
        next_token = torch.randint(0, model.config.vocab_size, (1, 1))
        full_sequence = torch.cat([prompt, next_token], dim=1)

        # Full recompute: no cache, whole growing sequence every time -
        # this is today's (pre-optimization) behavior.
        with torch.no_grad():
            hidden = shard1(full_sequence)
            full_logits = shard2(hidden)
        full_last_logits = full_logits[0, -1, :]

        # Cached: prefill the prompt once (fresh cache), then decode
        # only the single new token, reusing that cache.
        cache1 = shard1.new_cache()
        cache2 = shard2.new_cache()
        with torch.no_grad():
            hidden = shard1(prompt, cache_params=cache1)
            shard2(hidden, cache_params=cache2)
            hidden_next = shard1(next_token, cache_params=cache1)
            cached_logits = shard2(hidden_next, cache_params=cache2)
        cached_last_logits = cached_logits[0, -1, :]

        assert torch.allclose(full_last_logits, cached_last_logits, atol=1e-4)
