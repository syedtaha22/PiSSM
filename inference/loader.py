"""
Model loader for HuggingFace models.

Loads models and tokenizers from HuggingFace for inference on worker
nodes. Handles warning suppression, CPU-only enforcement, and model
residency management. Currently supports Mamba architecture only.
"""

import gc
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import psutil
import torch

import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*fast path.*")

from transformers import AutoTokenizer, MambaForCausalLM  # noqa: E402

from inference.manifest import ModelManifest  # noqa: E402

logger = logging.getLogger(__name__)

_ARCH_TO_MODEL_CLASS = {
    "mamba": MambaForCausalLM,
}


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
        The loaded tokenizer instance.
    manifest : ModelManifest
        The manifest used to load this model.
    memory_mb : int
        Approximate memory consumed by the model in megabytes.
    loaded_at : float
        Monotonic timestamp when the model was loaded.
    """

    name: str
    model: Any
    tokenizer: Any
    manifest: ModelManifest
    memory_mb: int
    loaded_at: float


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

    process = psutil.Process()
    mem_before = process.memory_info().rss

    logger.info("Loading tokenizer '%s'", manifest.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(manifest.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model '%s' (%s)", manifest.name, manifest.checkpoint)
    model = model_class.from_pretrained(manifest.checkpoint)
    model.to("cpu")
    model.eval()

    mem_after = process.memory_info().rss
    memory_mb = max(0, (mem_after - mem_before) // (1024 * 1024))

    logger.info(
        "Model '%s' loaded: ~%d MB, %d layers",
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
    process = psutil.Process()
    mem_before = process.memory_info().rss

    handle.model = None
    handle.tokenizer = None
    gc.collect()

    mem_after = process.memory_info().rss
    memory_freed = (mem_before - mem_after) // (1024 * 1024)

    logger.info("Model '%s' unloaded: ~%d MB freed", handle.name, memory_freed)
    return max(memory_freed, 0)


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
