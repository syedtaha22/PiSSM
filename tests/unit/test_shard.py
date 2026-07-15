"""
Tests for MambaShardModule.

Validates the shard module's forward pass shape for each pipeline
position (first, middle, last, single) using lightweight mock sub-modules
so no real model weights are required.
"""

import pytest
import torch
import torch.nn as nn

from inference.shard import MambaShardModule


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
            layers=[nn.Identity()],
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
            layers=[nn.Identity()],
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
            layers=[nn.Identity()],
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
            layers=[nn.Identity()],
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
            def forward(self, x):
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
