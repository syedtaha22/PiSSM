"""
Tests for tensor serialization and deserialization utilities.

Validates that PyTorch tensors survive a serialize-deserialize
roundtrip with correct values, shapes, and dtypes. These utilities
are used to transport tensors over gRPC as raw bytes.
"""

import pytest
import torch

from inference.tensor_utils import deserialize_tensor, serialize_tensor


class TestRoundtrip:
    """
    Tests that tensors survive a serialize-deserialize cycle.
    """

    def test_roundtrip_float32(self):
        """
        A float32 tensor survives roundtrip with correct values.
        """
        original = torch.tensor([1.0, 2.5, -3.7], dtype=torch.float32)
        data, shape, dtype_str = serialize_tensor(original)
        result = deserialize_tensor(data, shape, dtype_str)

        assert torch.equal(original, result)
        assert result.dtype == torch.float32

    def test_roundtrip_int64(self):
        """
        An int64 tensor survives roundtrip with correct values.
        Token IDs are int64.
        """
        original = torch.tensor([101, 2023, 3045, 0], dtype=torch.int64)
        data, shape, dtype_str = serialize_tensor(original)
        result = deserialize_tensor(data, shape, dtype_str)

        assert torch.equal(original, result)
        assert result.dtype == torch.int64

    def test_roundtrip_2d(self):
        """
        A 2D tensor (batch of token IDs) preserves shape through roundtrip.
        """
        original = torch.randint(0, 50000, (1, 128), dtype=torch.int64)
        data, shape, dtype_str = serialize_tensor(original)
        result = deserialize_tensor(data, shape, dtype_str)

        assert torch.equal(original, result)
        assert result.shape == (1, 128)

    def test_roundtrip_3d(self):
        """
        A 3D tensor (batch of activations) preserves shape through roundtrip.
        """
        original = torch.randn(1, 128, 768, dtype=torch.float32)
        data, shape, dtype_str = serialize_tensor(original)
        result = deserialize_tensor(data, shape, dtype_str)

        assert torch.allclose(original, result)
        assert result.shape == (1, 128, 768)


class TestErrors:
    """
    Tests for error handling in deserialization.
    """

    def test_unknown_dtype_raises(self):
        """
        An unsupported dtype string raises ValueError.
        """
        data = b"\x00" * 16
        with pytest.raises(ValueError, match="dtype"):
            deserialize_tensor(data, [4], "torch.bfloat16")

    def test_shape_mismatch_raises(self):
        """
        A shape that doesn't match the data size raises ValueError.
        """
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        data, _, dtype_str = serialize_tensor(original)

        with pytest.raises(ValueError):
            deserialize_tensor(data, [10], dtype_str)
