"""
In-memory registry of validated model manifests.

The orchestrator uses this to track which models have been submitted
and validated, and are available for inference dispatch. Thread-safe
for concurrent access from gRPC handler threads.
"""

import threading

from inference.manifest import ModelManifest


class ModelRegistry:
    """
    Thread-safe registry of model manifests.

    Stores validated ModelManifest instances keyed by model name.
    Follows the same locking pattern as the NodeRegistry.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelManifest] = {}
        self._lock = threading.Lock()

    def register(self, manifest: ModelManifest) -> None:
        """
        Register a validated manifest.

        Parameters
        ----------
        manifest : ModelManifest
            The manifest to register.

        Raises
        ------
        ValueError
            If a model with the same name is already registered.
        """
        with self._lock:
            if manifest.name in self._models:
                raise ValueError(f"Model '{manifest.name}' is already registered")
            self._models[manifest.name] = manifest

    def get(self, name: str) -> ModelManifest | None:
        """
        Look up a model by name.

        Parameters
        ----------
        name : str
            The model name to look up.

        Returns
        -------
        ModelManifest or None
            The manifest if found, None otherwise.
        """
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> list[ModelManifest]:
        """
        Return all registered models.

        Returns
        -------
        list[ModelManifest]
            All registered manifests.
        """
        with self._lock:
            return list(self._models.values())

    def delete(self, name: str) -> bool:
        """
        Remove a model by name.

        Parameters
        ----------
        name : str
            The model name to remove.

        Returns
        -------
        bool
            True if the model was found and removed, False otherwise.
        """
        with self._lock:
            if name in self._models:
                del self._models[name]
                return True
            return False
