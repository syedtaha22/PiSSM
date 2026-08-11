"""
Model loader for HuggingFace models.

Loads models and tokenizers from HuggingFace for inference on worker
nodes. Handles warning suppression, CPU-only enforcement, and model
residency management. Currently supports Mamba architecture only.
"""

import gc
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*fast path.*")

# suppress noisy third-party output from huggingface_hub:
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

from transformers import AutoTokenizer, MambaForCausalLM  # noqa: E402

from inference.manifest import ModelManifest  # noqa: E402

logger = logging.getLogger(__name__)

_ARCH_TO_MODEL_CLASS = {
    "mamba": MambaForCausalLM,
}


def _weight_bytes_mb(module: torch.nn.Module) -> int:
    """
    Sum of a module's parameter and buffer tensor byte sizes.

    Parameters
    ----------
    module : torch.nn.Module
        The model or shard to measure.

    Returns
    -------
    int
        Total parameter and buffer byte size in megabytes.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    return total_bytes // (1024 * 1024)


@dataclass
class ModelHandle:
    """
    A loaded model and its associated metadata.

    Parameters
    ----------
    name : str
        The model's registered name from the manifest.
    model : Any
        The loaded PyTorch model instance.
    tokenizer : Any
        The loaded tokenizer instance. None for pipeline shards.
    manifest : Any
        The manifest used to load this model. None for pipeline shards.
    memory_mb : int
        Byte size of the model's parameter and buffer tensors, in MB.
    loaded_at : float
        Monotonic timestamp when the model was loaded.
    layer_start : int
        First layer index owned by this shard. Zero for full models.
    layer_end : int
        Last layer index (exclusive) owned by this shard. Zero for full models.
    is_first_shard : bool
        True if this shard owns the embedding component.
    is_last_shard : bool
        True if this shard owns the final norm and lm_head.
    next_worker_address : str
        Address of the next worker in the pipeline, or empty string.
    cache : transformers.cache_utils.Cache or None
        Recurrent state (conv + SSM state per layer) for incremental
        decoding, carried across successive RunShard calls for the same
        generation. None when idle or between generations - a fresh
        one is built via model.new_cache() at the start of each new
        generation (see inference/service.py's RunShard handler).
    """

    name: str
    model: Any
    tokenizer: Any
    manifest: Any
    memory_mb: int
    loaded_at: float
    layer_start: int = 0
    layer_end: int = 0
    is_first_shard: bool = False
    is_last_shard: bool = False
    next_worker_address: str = ""
    cache: Any = None


def load_model(manifest: ModelManifest) -> ModelHandle:
    """
    Load a model and tokenizer from HuggingFace.

    Downloads the model (or loads from cache), sets it to CPU-only
    eval mode, and measures approximate memory usage.

    Parameters
    ----------
    manifest : ModelManifest
        The manifest describing the model to load.

    Returns
    -------
    ModelHandle
        The loaded model handle.

    Raises
    ------
    NotImplementedError
        If the manifest's architecture is not supported.
    """
    model_class = _ARCH_TO_MODEL_CLASS.get(manifest.arch)
    if model_class is None:
        raise NotImplementedError(
            f"Architecture '{manifest.arch}' is not supported. "
            f"Supported: {list(_ARCH_TO_MODEL_CLASS.keys())}"
        )

    logger.info("Loading tokenizer '%s'", manifest.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(manifest.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model '%s' (%s)", manifest.name, manifest.checkpoint)
    model = model_class.from_pretrained(manifest.checkpoint)
    model.to("cpu")
    model.eval()

    memory_mb = _weight_bytes_mb(model)

    logger.info(
        "Model '%s' ready: weights ~%d MB, %d layers",
        manifest.name,
        memory_mb,
        manifest.layers,
    )

    return ModelHandle(
        name=manifest.name,
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
        memory_mb=memory_mb,
        loaded_at=time.monotonic(),
    )


def unload_model(handle: ModelHandle) -> int:
    """
    Release a loaded model from memory.

    Deletes model and tokenizer references and triggers garbage
    collection.

    Parameters
    ----------
    handle : ModelHandle
        The model handle to unload.

    Returns
    -------
    int
        Approximate memory freed in megabytes.
    """
    memory_freed = handle.memory_mb

    handle.model = None
    handle.tokenizer = None
    gc.collect()

    # logger.info("Model '%s' unloaded: ~%d MB freed", handle.name, memory_freed)
    return memory_freed


def load_shard_from_metadata(
    tensor_locations: list,
    model_config_json_bytes: bytes,
    arch: str,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
    checkpoint: str,
    next_worker_address: str = "",
    model_name: str = "shard",
) -> ModelHandle:
    """
    Build a shard module directly for its layer range and load its weights.

    Workers call this after receiving a LoadShardRequest's metadata:
    it builds this shard's own layers/embedding/norm/lm_head directly
    from config, then fetches and loads only the tensors named in
    tensor_locations. The shard module is placed in CPU eval mode. No
    tokenizer is loaded.

    Parameters
    ----------
    tensor_locations : list[inference.shard.TensorLocationSpec]
        Which tensors this shard needs and where to fetch them from.
    model_config_json_bytes : bytes
        JSON-encoded model config from model.config.to_json_string.
    arch : str
        Architecture string used to select the shard class.
    layer_start : int
        First layer index owned by this shard (inclusive).
    layer_end : int
        Last layer index owned by this shard (exclusive).
    is_first : bool
        True if this shard owns the embedding component.
    is_last : bool
        True if this shard owns the final norm and lm_head.
    checkpoint : str
        HuggingFace repo id or local path, used to resolve each tensor
        location's source_file to a local path.
    next_worker_address : str
        Address of the next worker in the pipeline, or empty string.
    model_name : str
        Name to assign the handle.

    Returns
    -------
    ModelHandle
        Loaded shard handle with tokenizer and manifest set to None.

    Raises
    ------
    NotImplementedError
        If the architecture is not registered in the shard class registry.
    """
    from inference.shard import _ARCH_TO_SHARD_CLASS

    shard_cls = _ARCH_TO_SHARD_CLASS.get(arch)
    if shard_cls is None:
        raise NotImplementedError(
            f"Architecture '{arch}' is not supported. "
            f"Supported: {list(_ARCH_TO_SHARD_CLASS.keys())}"
        )

    config_cls = _ARCH_TO_MODEL_CLASS[arch].config_class
    config = config_cls.from_dict(json.loads(model_config_json_bytes.decode()))

    shard = shard_cls.from_tensor_locations(
        config,
        layer_start,
        layer_end,
        is_first,
        is_last,
        checkpoint,
        tensor_locations,
    )
    shard.to("cpu")
    shard.eval()

    memory_mb = _weight_bytes_mb(shard)

    logger.info(
        "Shard '%s' ready: layers [%d, %d), weights ~%d MB",
        model_name,
        layer_start,
        layer_end,
        memory_mb,
    )

    return ModelHandle(
        name=model_name,
        model=shard,
        tokenizer=None,
        manifest=None,
        memory_mb=memory_mb,
        loaded_at=time.monotonic(),
        layer_start=layer_start,
        layer_end=layer_end,
        is_first_shard=is_first,
        is_last_shard=is_last,
        next_worker_address=next_worker_address,
    )


def tokenize(handle: ModelHandle, text: str) -> torch.Tensor:
    """
    Tokenize a text string using the model's tokenizer.

    Parameters
    ----------
    handle : ModelHandle
        The loaded model handle.
    text : str
        The input text to tokenize.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of (input_ids, attention_mask), both 2D tensors
        of shape (1, seq_len).
    """
    inputs = handle.tokenizer(text, return_tensors="pt")
    return inputs.input_ids, inputs.attention_mask


def generate(
    handle: ModelHandle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 30,
) -> str:
    """
    Run autoregressive generation on the loaded model.

    Parameters
    ----------
    handle : ModelHandle
        The loaded model handle.
    input_ids : torch.Tensor
        Tokenized input as a 2D tensor of shape (1, seq_len).
    attention_mask : torch.Tensor or None
        Attention mask matching input_ids shape. If None, no mask
        is passed to the model.
    max_new_tokens : int
        Maximum number of new tokens to generate.

    Returns
    -------
    str
        The decoded output text.
    """
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        output = handle.model.generate(input_ids, **kwargs)
    return handle.tokenizer.decode(
        output[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
