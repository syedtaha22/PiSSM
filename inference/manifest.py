"""
Model manifest parsing and validation.

Provides the ModelManifest dataclass for representing a model's
configuration, and functions to load and validate manifest YAML files.
Validation checks all required fields, supported architecture and
input type values, and positive integer constraints.
"""

from dataclasses import dataclass

import yaml

SUPPORTED_ARCHITECTURES = ("mamba", "s4", "llm-transformer")
SUPPORTED_INPUT_TYPES = ("text", "timeseries", "audio")
REQUIRED_FIELDS = (
    "name",
    "arch",
    "checkpoint",
    "layers",
    "hidden_dim",
    "state_dim",
    "input_type",
    "tokenizer",
)


class ManifestError(Exception):
    """
    Raised when a model manifest is invalid or cannot be loaded.
    """

    pass


@dataclass(frozen=True)
class ModelManifest:
    """
    Immutable representation of a model manifest.

    Parameters
    ----------
    name : str
        Unique model name, no spaces.
    arch : str
        Model architecture: "mamba", "s4", or "llm-transformer".
    checkpoint : str
        HuggingFace model ID or local path to the checkpoint.
    layers : int
        Number of model layers.
    hidden_dim : int
        Hidden dimension size.
    state_dim : int
        State space dimension size.
    input_type : str
        Input data type: "text", "timeseries", or "audio".
    tokenizer : str
        HuggingFace tokenizer ID or local path.
    """

    name: str
    arch: str
    checkpoint: str
    layers: int
    hidden_dim: int
    state_dim: int
    input_type: str
    tokenizer: str


def validate_manifest(data: dict) -> None:
    """
    Validate a manifest data dictionary.

    Checks that all required fields are present, architecture and
    input type are supported, numeric fields are positive integers,
    and the name contains no spaces.

    Parameters
    ----------
    data : dict
        The parsed YAML data to validate.

    Raises
    ------
    ManifestError
        If any validation check fails.
    """
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ManifestError(f"Missing required field: {field}")

    if data["arch"] not in SUPPORTED_ARCHITECTURES:
        raise ManifestError(
            f"Unsupported arch: '{data['arch']}'. "
            f"Must be one of {SUPPORTED_ARCHITECTURES}"
        )

    if data["input_type"] not in SUPPORTED_INPUT_TYPES:
        raise ManifestError(
            f"Unsupported input_type: '{data['input_type']}'. "
            f"Must be one of {SUPPORTED_INPUT_TYPES}"
        )

    for field in ("layers", "hidden_dim", "state_dim"):
        value = data[field]
        if not isinstance(value, int) or value <= 0:
            raise ManifestError(
                f"Field '{field}' must be a positive integer, got {value}"
            )

    if " " in data["name"]:
        raise ManifestError(
            f"Field 'name' must not contain spaces, got '{data['name']}'"
        )


def load_manifest(path: str) -> ModelManifest:
    """
    Load and validate a model manifest from a YAML file.

    Parameters
    ----------
    path : str
        Path to the manifest YAML file.

    Returns
    -------
    ModelManifest
        The validated manifest.

    Raises
    ------
    ManifestError
        If the file is not found or validation fails.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ManifestError(f"Manifest file not found: {path}")

    validate_manifest(data)

    return ModelManifest(
        name=data["name"],
        arch=data["arch"],
        checkpoint=data["checkpoint"],
        layers=data["layers"],
        hidden_dim=data["hidden_dim"],
        state_dim=data["state_dim"],
        input_type=data["input_type"],
        tokenizer=data["tokenizer"],
    )
