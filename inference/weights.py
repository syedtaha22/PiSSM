"""
Checkpoint metadata reading and file resolution.

read_checkpoint_metadata reads a checkpoint's safetensors header/index -
the orchestrator uses it to compute shard tensor-name assignments.
fetch_source_file resolves one checkpoint file to a local path,
downloading it if necessary - workers use it to pull only the file(s)
their assigned tensors live in. Both accept the same dual checkpoint
form used everywhere else in this project: an HF repo id, or a local
filesystem directory.
"""

import logging
import os
import warnings
from pathlib import Path

import tqdm as tqdm_module
from huggingface_hub import (
    get_local_safetensors_metadata,
    get_safetensors_metadata,
    hf_hub_download,
)

# suppress noisy third-party output from huggingface_hub:
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


class _LeaveFalseTqdm(tqdm_module.tqdm):
    """
    Makes tqdm progress bars disappear after completion, rather than
    leaving a static bar on the console.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("leave", False)
        super().__init__(*args, **kwargs)


def read_checkpoint_metadata(checkpoint: str):
    """
    Read a checkpoint's safetensors header/index.

    Parameters
    ----------
    checkpoint : str
        HuggingFace repo id or local filesystem path to the checkpoint.

    Returns
    -------
    huggingface_hub.SafetensorsRepoMetadata
        Carries .weight_map (tensor name -> source filename) and
        .files_metadata (filename -> per-tensor dtype/shape/offsets),
        read from the checkpoint's header/index alone.
    """
    if os.path.isdir(checkpoint):
        return get_local_safetensors_metadata(checkpoint)
    return get_safetensors_metadata(checkpoint)


def fetch_source_file(checkpoint: str, filename: str) -> Path:
    """
    Resolve one checkpoint file to a local path, downloading if needed.

    Parameters
    ----------
    checkpoint : str
        HuggingFace repo id or local filesystem path to the checkpoint.
    filename : str
        Name of the file within the checkpoint, e.g. "model.safetensors".

    Returns
    -------
    pathlib.Path
        Local path to the file: joined directly under checkpoint for a
        local checkpoint, or the huggingface_hub cache path after
        downloading for a repo id.
    """
    if os.path.isdir(checkpoint):
        return Path(checkpoint) / filename
    return Path(
        hf_hub_download(
            repo_id=checkpoint, filename=filename, tqdm_class=_LeaveFalseTqdm
        )
    )
