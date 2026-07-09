"""
Tests for the in-memory model registry.

Covers FR-MR-01 (registry storage), FR-MR-04 (listing),
FR-MR-05 (deletion), and NFR-02 (thread safety).
No torch dependency.
"""

import threading

import pytest

from inference.manifest import ModelManifest
from inference.model_registry import ModelRegistry


def make_manifest(name="mamba-130m"):
    """
    Create a ModelManifest with sensible defaults.

    Parameters
    ----------
    name : str
        Model name for the manifest.

    Returns
    -------
    ModelManifest
        A valid manifest instance.
    """
    return ModelManifest(
        name=name,
        arch="mamba",
        checkpoint="state-spaces/mamba-130m-hf",
        layers=24,
        hidden_dim=768,
        state_dim=16,
        input_type="text",
        tokenizer="EleutherAI/gpt-neox-20b",
    )


class TestRegisterAndGet:
    """
    Tests for registering and retrieving models.
    """

    def test_register_and_get(self):
        """
        A registered model can be retrieved by name.
        """
        registry = ModelRegistry()
        manifest = make_manifest()
        registry.register(manifest)

        result = registry.get("mamba-130m")
        assert result is not None
        assert result.name == "mamba-130m"
        assert result.arch == "mamba"

    def test_register_duplicate_raises(self):
        """
        Registering the same model name twice raises ValueError.
        """
        registry = ModelRegistry()
        manifest = make_manifest()
        registry.register(manifest)

        with pytest.raises(ValueError, match="mamba-130m"):
            registry.register(manifest)

    def test_get_unknown_returns_none(self):
        """
        Looking up an unregistered model returns None.
        """
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None


class TestListModels:
    """
    Tests for listing registered models.
    """

    def test_list_models_empty(self):
        """
        An empty registry returns an empty list.
        """
        registry = ModelRegistry()
        assert registry.list_models() == []

    def test_list_models_returns_all(self):
        """
        All registered models are returned.
        """
        registry = ModelRegistry()
        registry.register(make_manifest("model-a"))
        registry.register(make_manifest("model-b"))
        registry.register(make_manifest("model-c"))

        models = registry.list_models()
        assert len(models) == 3
        names = {m.name for m in models}
        assert names == {"model-a", "model-b", "model-c"}


class TestDeleteModel:
    """
    Tests for deleting models from the registry.
    """

    def test_delete_existing(self):
        """
        Deleting an existing model returns True and removes it.
        """
        registry = ModelRegistry()
        registry.register(make_manifest())

        assert registry.delete("mamba-130m") is True
        assert registry.get("mamba-130m") is None

    def test_delete_nonexistent(self):
        """
        Deleting a nonexistent model returns False.
        """
        registry = ModelRegistry()
        assert registry.delete("nonexistent") is False


class TestThreadSafety:
    """
    Verify concurrent access does not corrupt registry state.
    """

    def test_concurrent_register(self):
        """
        Multiple threads registering different models simultaneously
        should not raise exceptions or lose entries.
        """
        registry = ModelRegistry()
        errors = []

        def register_loop(name):
            try:
                registry.register(make_manifest(name))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_loop, args=(f"model-{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(registry.list_models()) == 20
