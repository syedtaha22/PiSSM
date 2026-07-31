"""
Tests for orchestrator.server helper functions.

Covers FR-MR-01/03 (manifest auto-registration at startup). No gRPC
or HTTP server involved - these call register_existing_manifests
directly against a real ModelRegistry and a temporary directory.
"""

from inference.model_registry import ModelRegistry
from orchestrator.server import register_existing_manifests

VALID_MANIFEST_YAML = """
name: mamba-130m
arch: mamba
checkpoint: state-spaces/mamba-130m-hf
layers: 24
hidden_dim: 768
state_dim: 16
input_type: text
tokenizer: EleutherAI/gpt-neox-20b
"""


class TestRegisterExistingManifests:
    """
    Tests for register_existing_manifests.
    """

    def test_registers_every_valid_manifest_in_directory(self, tmp_path):
        """
        Every valid *.yaml file in the directory is registered.
        """
        (tmp_path / "mamba-130m.yaml").write_text(VALID_MANIFEST_YAML)
        model_registry = ModelRegistry()

        register_existing_manifests(model_registry, tmp_path)

        assert model_registry.get("mamba-130m") is not None

    def test_skips_invalid_manifest_without_raising(self, tmp_path):
        """
        An invalid manifest is logged and skipped, not raised.
        """
        (tmp_path / "broken.yaml").write_text("name: broken\narch: mamba\n")
        model_registry = ModelRegistry()

        register_existing_manifests(model_registry, tmp_path)

        assert model_registry.list_models() == []

    def test_missing_directory_does_nothing(self, tmp_path):
        """
        A nonexistent manifests directory is a no-op, not an error.
        """
        model_registry = ModelRegistry()

        register_existing_manifests(model_registry, tmp_path / "does-not-exist")

        assert model_registry.list_models() == []

    def test_ignores_non_yaml_files(self, tmp_path):
        """
        Non-.yaml files in the directory are not treated as manifests.
        """
        (tmp_path / "README.md").write_text("not a manifest")
        model_registry = ModelRegistry()

        register_existing_manifests(model_registry, tmp_path)

        assert model_registry.list_models() == []
