"""
Checkpoint metadata host for shard planning on the orchestrator.

The orchestrator calls load() once to read a checkpoint's safetensors
header/index and model config, then calls extract_shard() for each
assignment in the dispatch plan to get the list of tensors that shard
owns and where to fetch them from. Neither step touches the checkpoint's
weight bytes, so this class's memory footprint is O(metadata size).
"""

from huggingface_hub.utils import disable_progress_bars
from transformers import AutoConfig

from inference.shard import _ARCH_TO_SHARD_CLASS
from inference.weights import read_checkpoint_metadata

# disable huggingface's progress bars on the orchestrator
disable_progress_bars()


class ModelStore:
    """
    Reads checkpoint metadata and computes per-shard tensor assignments.

    The orchestrator calls load() once, then calls extract_shard() for
    each assignment in the dispatch plan to obtain (tensor_locations,
    config_json_bytes) to send in LoadShard requests. Call unload() to
    drop the cached metadata once all shards are distributed.
    """

    def __init__(self) -> None:
        self._metadata = None
        self._config_json = None

    def load(self, manifest) -> None:
        """
        Read the checkpoint's safetensors header/index and model config.

        Parameters
        ----------
        manifest : ModelManifest
            The manifest describing the model to load.
        """
        self._metadata = read_checkpoint_metadata(manifest.checkpoint)
        config = AutoConfig.from_pretrained(manifest.checkpoint)
        # Force the manifest's declared dtype rather than trust whatever
        # (if anything) the checkpoint's own config.json happens to carry -
        # this is what workers build shard layers in directly, so it must
        # match the checkpoint's real on-disk dtype exactly.
        config.dtype = manifest.dtype
        self._config_json = config.to_json_string().encode()

    def extract_shard(
        self,
        arch: str,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ) -> tuple[list, bytes]:
        """
        Resolve a shard's tensor locations and config for a layer range.

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
        tuple[list, bytes]
            (tensor_locations, model_config_json_bytes), where
            tensor_locations is a list of the arch's TensorLocationSpec.

        Raises
        ------
        RuntimeError
            If no checkpoint metadata is loaded.
        NotImplementedError
            If the architecture is not registered in _ARCH_TO_SHARD_CLASS.
        """
        if self._metadata is None:
            raise RuntimeError("No checkpoint metadata loaded. Call load() first.")

        shard_cls = _ARCH_TO_SHARD_CLASS.get(arch)
        if shard_cls is None:
            raise NotImplementedError(
                f"Architecture '{arch}' is not supported. "
                f"Supported: {list(_ARCH_TO_SHARD_CLASS.keys())}"
            )

        locations = shard_cls.tensor_locations_for_range(
            self._metadata.weight_map, layer_start, layer_end, is_first, is_last
        )

        return locations, self._config_json

    def unload(self) -> None:
        """
        Release the cached checkpoint metadata.
        """
        self._metadata = None
        self._config_json = None
