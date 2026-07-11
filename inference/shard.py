"""
Architecture-specific shard modules for pipeline-parallel inference.

Each shard module holds a contiguous slice of a model's layer stack,
plus optional embedding (first shard) and norm + lm_head (last shard).
New architectures register a class in _ARCH_TO_SHARD_CLASS.
"""

import gc
import io
import json

import torch
import torch.nn as nn


class MambaShardModule(nn.Module):
    """
    A contiguous slice of a MambaForCausalLM for pipeline-parallel use.

    The first shard embeds token IDs before running the assigned layers.
    The last shard applies the final norm and language model head after
    the assigned layers. Middle shards pass hidden states through unchanged.

    Parameters
    ----------
    layers : list[nn.Module]
        The MambaBlock instances for the assigned layer range.
    is_first : bool
        True if this shard owns the token embedding.
    is_last : bool
        True if this shard owns the final norm and language model head.
    embeddings : nn.Module or None
        Token embedding module. Required when is_first is True.
    norm_f : nn.Module or None
        Final layer norm. Required when is_last is True.
    lm_head : nn.Module or None
        Language model head projection. Required when is_last is True.
    """

    def __init__(
        self,
        layers: list,
        is_first: bool,
        is_last: bool,
        embeddings: nn.Module | None = None,
        norm_f: nn.Module | None = None,
        lm_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        if is_first:
            if embeddings is None:
                raise ValueError("embeddings required when is_first=True")
            self.embeddings = embeddings
        self.layers = nn.ModuleList(layers)
        if is_last:
            if norm_f is None or lm_head is None:
                raise ValueError("norm_f and lm_head required when is_last=True")
            self.norm_f = norm_f
            self.lm_head = lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the shard forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Token IDs when is_first is True, hidden states otherwise.

        Returns
        -------
        torch.Tensor
            Hidden states for non-last shards, logits for the last shard.
        """
        if self.is_first:
            x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        if self.is_last:
            x = self.norm_f(x)
            x = self.lm_head(x)
        return x

    @classmethod
    def from_model(
        cls,
        model,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ) -> "MambaShardModule":
        """
        Extract a shard from a full MambaForCausalLM model.

        Parameters
        ----------
        model : MambaForCausalLM
            The source model to slice.
        layer_start : int
            First layer index (inclusive).
        layer_end : int
            Last layer index (exclusive).
        is_first : bool
            True if this shard owns the embedding.
        is_last : bool
            True if this shard owns norm and lm_head.

        Returns
        -------
        MambaShardModule
            A shard wrapping the assigned layer slice.
        """
        layers = list(model.backbone.layers[layer_start:layer_end])
        embeddings = model.backbone.embeddings if is_first else None
        norm_f = model.backbone.norm_f if is_last else None
        lm_head = model.lm_head if is_last else None
        return cls(layers, is_first, is_last, embeddings, norm_f, lm_head)

    @classmethod
    def from_bytes(
        cls,
        weights_bytes: bytes,
        config_json_bytes: bytes,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ) -> "MambaShardModule":
        """
        Reconstruct a shard from serialized weights and config.

        Creates a temporary full model from the received config to
        establish the correct layer architecture, slices the target
        range, loads the received state_dict, and discards the full
        model reference.

        Parameters
        ----------
        weights_bytes : bytes
            Serialized shard state_dict from torch.save.
        config_json_bytes : bytes
            JSON-encoded model config from model.config.to_json_string.
        layer_start : int
            First layer index (inclusive).
        layer_end : int
            Last layer index (exclusive).
        is_first : bool
            True if this shard owns the embedding.
        is_last : bool
            True if this shard owns norm and lm_head.

        Returns
        -------
        MambaShardModule
            A shard with weights loaded from the received bytes.
        """
        from transformers import MambaConfig, MambaForCausalLM

        config = MambaConfig.from_dict(json.loads(config_json_bytes.decode()))
        tmp = MambaForCausalLM(config)
        shard = cls.from_model(tmp, layer_start, layer_end, is_first, is_last)
        del tmp
        gc.collect()

        state_dict = torch.load(io.BytesIO(weights_bytes), weights_only=True)
        shard.load_state_dict(state_dict)
        return shard


_ARCH_TO_SHARD_CLASS: dict[str, type] = {
    "mamba": MambaShardModule,
}
