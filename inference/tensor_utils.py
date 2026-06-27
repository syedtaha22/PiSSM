"""
Tensor serialization and deserialization for gRPC transport.

Converts PyTorch tensors to raw bytes with shape and dtype metadata
for transmission over protobuf, and reconstructs them on the other
side. No pickle - tensors are transported as contiguous numpy buffers.
"""

import numpy as np
import torch

DTYPE_MAP = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
}


def serialize_tensor(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    """
    Serialize a PyTorch tensor to raw bytes with metadata.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to serialize.

    Returns
    -------
    tuple[bytes, list[int], str]
        A tuple of (raw bytes, shape as list of ints, dtype string).
    """
    array = tensor.detach().contiguous().numpy()
    return array.tobytes(), list(tensor.shape), str(tensor.dtype)


def deserialize_tensor(data: bytes, shape: list[int], dtype_str: str) -> torch.Tensor:
    """
    Reconstruct a PyTorch tensor from raw bytes and metadata.

    Parameters
    ----------
    data : bytes
        Raw byte buffer from serialize_tensor.
    shape : list[int]
        Shape of the original tensor.
    dtype_str : str
        Dtype string, e.g. "torch.float32", "torch.int64".

    Returns
    -------
    torch.Tensor
        The reconstructed tensor.

    Raises
    ------
    ValueError
        If the dtype string is not supported or the data size
        does not match the given shape and dtype.
    """
    if dtype_str not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype: '{dtype_str}'. "
            f"Must be one of {list(DTYPE_MAP.keys())}"
        )

    np_dtype = DTYPE_MAP[dtype_str]
    expected_size = int(np.prod(shape)) * np.dtype(np_dtype).itemsize

    if len(data) != expected_size:
        raise ValueError(
            f"Data size {len(data)} does not match shape {shape} "
            f"with dtype {dtype_str} (expected {expected_size} bytes)"
        )

    array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(array.copy())
