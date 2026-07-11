"""
Full-model host for weight extraction on the orchestrator.

The orchestrator loads the complete model once, extracts per-shard
weight bytes for each worker in the dispatch plan, and drops the full
model reference after all shards are distributed.
"""

import gc
import io

import torch

from inference.loader import load_model, unload_model
from inference.shard import _ARCH_TO_SHARD_CLASS


class ModelStore:
    """
    Downloads a full model and extracts shard weight bytes for workers.

    The orchestrator calls load() once, then calls extract_shard() for
    each assignment in the dispatch plan to obtain (weights_bytes,
    config_json_bytes) to send in LoadShard requests. Call unload() to
    release the full model from memory after all shards are distributed.
    """

    def __init__(self) -> None:
        self._handle = None

    def load(self, manifest) -> None:
        """
        Download and load the full model into orchestrator memory.

        Parameters
        ----------
        manifest : ModelManifest
            The manifest describing the model to load.
        """
        self._handle = load_model(manifest)

    def extract_shard(
        self,
        arch: str,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ) -> tuple[bytes, bytes]:
        """
        Serialize a shard's weights and config for a given layer range.

        Parameters
        ----------
        arch : str
            Architecture string used to look up the shard class.
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
        tuple[bytes, bytes]
            (shard_weights_bytes, model_config_json_bytes)

        Raises
        ------
        RuntimeError
            If no model is loaded.
        NotImplementedError
            If the architecture is not registered in _ARCH_TO_SHARD_CLASS.
        """
        if self._handle is None:
            raise RuntimeError("No model loaded. Call load() first.")

        shard_cls = _ARCH_TO_SHARD_CLASS.get(arch)
        if shard_cls is None:
            raise NotImplementedError(
                f"Architecture '{arch}' is not supported. "
                f"Supported: {list(_ARCH_TO_SHARD_CLASS.keys())}"
            )

        shard = shard_cls.from_model(
            self._handle.model, layer_start, layer_end, is_first, is_last
        )

        buf = io.BytesIO()
        torch.save(shard.state_dict(), buf)
        weights_bytes = buf.getvalue()

        config_json = self._handle.model.config.to_json_string().encode()

        return weights_bytes, config_json

    def unload(self) -> None:
        """
        Release the loaded model from orchestrator memory.
        """
        if self._handle is not None:
            unload_model(self._handle)
            self._handle = None
            gc.collect()
