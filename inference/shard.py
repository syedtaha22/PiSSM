"""
Architecture-specific shard modules for pipeline-parallel inference.

Each shard module holds a contiguous slice of a model's layer stack,
plus optional embedding (first shard) and norm + lm_head (last shard).
New architectures register a class in _ARCH_TO_SHARD_CLASS.
"""

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mamba.modeling_mamba import MambaBlock, MambaRMSNorm

from inference.weights import fetch_source_file

_LAYER_NAME_RE = re.compile(r"^backbone\.layers\.(\d+)\.(.+)$")


@dataclass
class TensorLocationSpec:
    """
    One tensor a shard needs and where to fetch it from.

    Plain-Python mirror of the proto TensorLocation message, kept
    independent of protobuf so tensor_locations_for_range stays a pure,
    trivially unit-testable function.

    Parameters
    ----------
    dest_name : str
        The parameter's name within the shard module's own state_dict,
        e.g. "layers.0.mixer.A_log" or "lm_head.weight".
    source_name : str
        The tensor's name within the checkpoint's weight map, e.g.
        "backbone.layers.5.mixer.A_log". Differs from dest_name only
        for a tied lm_head, which sources from the embedding tensor.
    source_file : str
        The checkpoint file this tensor lives in.
    """

    dest_name: str
    source_name: str
    source_file: str


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
    config : transformers.MambaConfig or None
        The *full* (un-sliced) model's config - not this shard's own
        layer count. Needed to size a correctly-shaped recurrent-state
        cache via new_cache(): each MambaBlock retains its original,
        global layer_idx after slicing, and Cache.layers[layer_idx]
        indexes positionally, so a cache sized to this shard's own
        (smaller) layer count would raise IndexError the moment a
        non-zero-based layer_idx is used.
    """

    def __init__(
        self,
        layers: list,
        is_first: bool,
        is_last: bool,
        embeddings: nn.Module | None = None,
        norm_f: nn.Module | None = None,
        lm_head: nn.Module | None = None,
        config=None,
    ) -> None:
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.config = config
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

    def forward(
        self, x: torch.Tensor, cache_params: Cache | None = None
    ) -> torch.Tensor:
        """
        Run the shard forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Token IDs when is_first is True, hidden states otherwise.
            The full sequence for a prefill call, or just the single
            newest token/timestep for a cached decode call.
        cache_params : transformers.cache_utils.Cache or None
            Recurrent state (conv + SSM state per layer) carried across
            calls for incremental decoding. None runs a stateless full
            pass, same as before this feature existed.

        Returns
        -------
        torch.Tensor
            Hidden states for non-last shards, logits for the last shard.
        """
        if self.is_first:
            x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, cache_params=cache_params)
        if self.is_last:
            x = self.norm_f(x)
            x = self.lm_head(x)
        return x

    def new_cache(self) -> Cache:
        """
        Build a fresh recurrent-state cache sized for this shard.

        Always sized from the full model's config
        (self.config.num_hidden_layers), never from this shard's own
        (smaller) layer count - see the class docstring for why.

        Returns
        -------
        transformers.cache_utils.Cache
            An empty cache ready for a prefill call.
        """
        return DynamicCache(config=self.config)

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
        return cls(
            layers, is_first, is_last, embeddings, norm_f, lm_head, config=model.config
        )

    @classmethod
    def tensor_locations_for_range(
        cls,
        weight_map: dict,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ) -> list:
        """
        Resolve which checkpoint tensors this shard needs and where.

        Pure name-matching against a checkpoint's weight_map - no
        network or tensor bytes involved. Layer tensor names are
        renumbered from the checkpoint's global layer index to this
        shard's own local index (layer_idx - layer_start), since the
        shard's own ModuleList is always zero-based.

        Parameters
        ----------
        weight_map : dict
            Tensor name -> source filename, as read from the
            checkpoint's safetensors header/index
            (inference.weights.read_checkpoint_metadata).
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
        list[TensorLocationSpec]
            One entry per tensor this shard needs to load.
        """
        locations = []

        if is_first:
            name = "backbone.embeddings.weight"
            locations.append(
                TensorLocationSpec(
                    dest_name="embeddings.weight",
                    source_name=name,
                    source_file=weight_map[name],
                )
            )

        for name, source_file in weight_map.items():
            match = _LAYER_NAME_RE.match(name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            if not (layer_start <= layer_idx < layer_end):
                continue
            local_idx = layer_idx - layer_start
            locations.append(
                TensorLocationSpec(
                    dest_name=f"layers.{local_idx}.{match.group(2)}",
                    source_name=name,
                    source_file=source_file,
                )
            )

        if is_last:
            norm_name = "backbone.norm_f.weight"
            locations.append(
                TensorLocationSpec(
                    dest_name="norm_f.weight",
                    source_name=norm_name,
                    source_file=weight_map[norm_name],
                )
            )

            lm_head_name = "lm_head.weight"
            if lm_head_name in weight_map:
                source_name = lm_head_name
            else:
                # Tied to the input embedding (mamba's default): the
                # checkpoint has no separate storage for lm_head.weight,
                # so load it from the embedding tensor instead.
                source_name = "backbone.embeddings.weight"
            locations.append(
                TensorLocationSpec(
                    dest_name="lm_head.weight",
                    source_name=source_name,
                    source_file=weight_map[source_name],
                )
            )

        return locations

    @classmethod
    def from_tensor_locations(
        cls,
        config,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
        checkpoint: str,
        tensor_locations: list,
    ) -> "MambaShardModule":
        """
        Build a shard directly for [layer_start, layer_end) and load it.

        Constructs only this shard's own layers, embedding, and/or
        norm+lm_head straight from config, then fetches each distinct
        checkpoint file tensor_locations references (once per file) and
        pulls out just the needed tensors via safetensors' mmap-backed
        get_tensor(), which reads each tensor from disk on demand.

        Parameters
        ----------
        config : transformers.MambaConfig
            The *full* (un-sliced) model's config.
        layer_start : int
            First layer index (inclusive).
        layer_end : int
            Last layer index (exclusive).
        is_first : bool
            True if this shard owns the embedding.
        is_last : bool
            True if this shard owns norm and lm_head.
        checkpoint : str
            HuggingFace repo id or local path, used to resolve each
            source_file in tensor_locations to a local path.
        tensor_locations : list[TensorLocationSpec]
            Which tensors to load and where each comes from.

        Returns
        -------
        MambaShardModule
            A shard built for this layer range with the requested
            tensors loaded.
        """
        layers = [
            MambaBlock(config, layer_idx=i) for i in range(layer_start, layer_end)
        ]
        embeddings = (
            nn.Embedding(config.vocab_size, config.hidden_size) if is_first else None
        )
        norm_f = (
            MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            if is_last
            else None
        )
        lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if is_last
            else None
        )
        shard = cls(
            layers, is_first, is_last, embeddings, norm_f, lm_head, config=config
        )

        locations_by_file: dict[str, list] = {}
        for location in tensor_locations:
            locations_by_file.setdefault(location.source_file, []).append(location)

        state_dict = {}
        for source_file, locations in locations_by_file.items():
            path = fetch_source_file(checkpoint, source_file)
            with safe_open(path, framework="pt") as f:
                for location in locations:
                    state_dict[location.dest_name] = f.get_tensor(location.source_name)

        shard.load_state_dict(state_dict)
        return shard


_ARCH_TO_SHARD_CLASS: dict[str, type] = {
    "mamba": MambaShardModule,
}
