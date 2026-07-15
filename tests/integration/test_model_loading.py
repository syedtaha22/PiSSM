"""
Integration tests for model loading and inference.

These tests download and load the actual Mamba-130M model from
HuggingFace. They are slow (model download on first run) and
marked with @pytest.mark.slow.
"""

import subprocess
import sys

import pytest
import torch

from inference.manifest import load_manifest
from inference.loader import generate, load_model, tokenize, unload_model
from inference.shard import MambaShardModule

MANIFEST_PATH = "manifests/mamba-130m.yaml"
REFERENCE_PROMPT = "Hey how are you doing?"
REFERENCE_OUTPUT = (
    "Hey how are you doing?\n\n"
    "I'm so glad you're here. "
    "I'm so glad you're here. "
    "I'm so glad you're here. "
    "I'm so glad"
)


@pytest.fixture(scope="module")
def model_handle():
    """
    Load Mamba-130M once for the entire test module.

    Yields
    ------
    ModelHandle
        The loaded model handle.
    """
    manifest = load_manifest(MANIFEST_PATH)
    handle = load_model(manifest)
    yield handle
    unload_model(handle)


@pytest.mark.slow
class TestModelLoading:
    """
    Tests that the model loads correctly from HuggingFace.
    """

    def test_model_handle_fields(self, model_handle):
        """
        The loaded handle has correct name, non-None model and tokenizer,
        and positive memory usage.
        """
        assert model_handle.name == "mamba-130m"
        assert model_handle.model is not None
        assert model_handle.tokenizer is not None
        assert model_handle.memory_mb >= 0
        assert model_handle.loaded_at > 0


@pytest.mark.slow
class TestEndToEndInference:
    """
    Tests that tokenization and generation produce correct output.
    """

    def test_tokenize_returns_tensors(self, model_handle):
        """
        Tokenizing a string returns input_ids and attention_mask,
        both 2D int64 tensors.
        """
        import torch

        ids, mask = tokenize(model_handle, REFERENCE_PROMPT)
        assert ids.dim() == 2
        assert ids.shape[0] == 1
        assert ids.shape[1] > 0
        assert ids.dtype == torch.int64
        assert mask.shape == ids.shape

    def test_generate_returns_string(self, model_handle):
        """
        Generation returns a non-empty string.
        """
        ids, mask = tokenize(model_handle, REFERENCE_PROMPT)
        output = generate(model_handle, ids, attention_mask=mask, max_new_tokens=10)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_reference_output_match(self, model_handle):
        """
        The same prompt with do_sample=False produces deterministic
        output matching the reference.
        """
        ids, mask = tokenize(model_handle, REFERENCE_PROMPT)
        output = generate(model_handle, ids, attention_mask=mask, max_new_tokens=30)
        assert output == REFERENCE_OUTPUT

    def test_no_warnings_on_stderr(self):
        """
        Running inference through the loader produces no warning
        messages on stderr. Progress bars are expected and allowed.
        """
        script = (
            "from inference.manifest import load_manifest; "
            "from inference.loader import load_model, tokenize, generate; "
            f"m = load_manifest('{MANIFEST_PATH}'); "
            "h = load_model(m); "
            f"ids, mask = tokenize(h, '{REFERENCE_PROMPT}'); "
            "generate(h, ids, attention_mask=mask, max_new_tokens=5)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        warning_lines = [
            line
            for line in result.stderr.splitlines()
            if "Warning" in line or "warning" in line or "ERROR" in line
        ]
        assert warning_lines == [], f"Unexpected warnings: {warning_lines}"


@pytest.mark.slow
class TestShardRoundtrip:
    """
    Tests that shard extraction and reconstruction preserve model output.

    Extracts two shards from a loaded Mamba-130M, reconstructs each from
    bytes, and verifies that running them in sequence produces the same
    logits as a forward pass through the full model.
    """

    def test_two_shard_roundtrip_matches_full_model(self, model_handle):
        """
        Extracting two shards, serializing, reconstructing, and running
        them in pipeline order produces logits matching the full model.
        """
        import io

        model = model_handle.model
        total_layers = len(model.backbone.layers)
        mid = total_layers // 2

        shard0 = MambaShardModule.from_model(
            model, 0, mid, is_first=True, is_last=False
        )
        shard1 = MambaShardModule.from_model(
            model, mid, total_layers, is_first=False, is_last=True
        )

        def roundtrip(shard, layer_start, layer_end, is_first, is_last):
            buf = io.BytesIO()
            torch.save(shard.state_dict(), buf)
            weights_bytes = buf.getvalue()
            config_json = model.config.to_json_string().encode()
            return MambaShardModule.from_bytes(
                weights_bytes, config_json, layer_start, layer_end, is_first, is_last
            )

        reconstructed0 = roundtrip(shard0, 0, mid, True, False)
        reconstructed1 = roundtrip(shard1, mid, total_layers, False, True)

        reconstructed0.eval()
        reconstructed1.eval()
        model.eval()

        input_ids = torch.tensor([[101, 2023, 3045]], dtype=torch.int64)

        with torch.no_grad():
            full_logits = model(input_ids).logits
            hidden = reconstructed0(input_ids)
            pipeline_logits = reconstructed1(hidden)

        assert pipeline_logits.shape == full_logits.shape
        assert torch.allclose(pipeline_logits, full_logits, atol=1e-5)
