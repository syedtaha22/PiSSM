"""
Tests for model manifest parsing and validation.

Covers FR-MR-01 (manifest storage), FR-MR-03 (validation with
clear error messages). Uses temporary YAML files via pytest's
tmp_path fixture. No torch dependency.
"""

import pytest
import yaml

from inference.manifest import ManifestError, load_manifest

VALID_MANIFEST_DATA = {
    "name": "mamba-130m",
    "arch": "mamba",
    "checkpoint": "state-spaces/mamba-130m-hf",
    "layers": 24,
    "hidden_dim": 768,
    "state_dim": 16,
    "input_type": "text",
    "tokenizer": "EleutherAI/gpt-neox-20b",
}


def write_manifest(tmp_path, data):
    """
    Write a manifest dict to a temporary YAML file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    data : dict
        Manifest data to write.

    Returns
    -------
    str
        Path to the written YAML file.
    """
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


class TestLoadValidManifest:
    """
    Tests for successfully loading a valid manifest.
    """

    def test_load_valid_manifest(self, tmp_path):
        """
        A valid manifest YAML produces a ModelManifest with all fields set.
        """
        path = write_manifest(tmp_path, VALID_MANIFEST_DATA)
        manifest = load_manifest(path)

        assert manifest.name == "mamba-130m"
        assert manifest.arch == "mamba"
        assert manifest.checkpoint == "state-spaces/mamba-130m-hf"
        assert manifest.layers == 24
        assert manifest.hidden_dim == 768
        assert manifest.state_dim == 16
        assert manifest.input_type == "text"
        assert manifest.tokenizer == "EleutherAI/gpt-neox-20b"

    def test_manifest_is_frozen(self, tmp_path):
        """
        ModelManifest instances are immutable.
        """
        path = write_manifest(tmp_path, VALID_MANIFEST_DATA)
        manifest = load_manifest(path)

        with pytest.raises(AttributeError):
            manifest.name = "something-else"


class TestMissingFields:
    """
    Tests for missing required fields in the manifest.
    """

    @pytest.mark.parametrize(
        "field",
        [
            "name",
            "arch",
            "checkpoint",
            "layers",
            "hidden_dim",
            "state_dim",
            "input_type",
            "tokenizer",
        ],
    )
    def test_missing_required_field(self, tmp_path, field):
        """
        Each missing required field raises ManifestError with the field name.
        """
        data = VALID_MANIFEST_DATA.copy()
        del data[field]
        path = write_manifest(tmp_path, data)

        with pytest.raises(ManifestError, match=field):
            load_manifest(path)


class TestInvalidValues:
    """
    Tests for invalid field values in the manifest.
    """

    def test_unsupported_arch(self, tmp_path):
        """
        An unsupported architecture raises ManifestError.
        """
        data = {**VALID_MANIFEST_DATA, "arch": "rnn"}
        path = write_manifest(tmp_path, data)

        with pytest.raises(ManifestError, match="arch"):
            load_manifest(path)

    def test_unsupported_input_type(self, tmp_path):
        """
        An unsupported input type raises ManifestError.
        """
        data = {**VALID_MANIFEST_DATA, "input_type": "video"}
        path = write_manifest(tmp_path, data)

        with pytest.raises(ManifestError, match="input_type"):
            load_manifest(path)

    def test_layers_zero(self, tmp_path):
        """
        Zero layers raises ManifestError.
        """
        data = {**VALID_MANIFEST_DATA, "layers": 0}
        path = write_manifest(tmp_path, data)

        with pytest.raises(ManifestError, match="layers"):
            load_manifest(path)

    def test_layers_negative(self, tmp_path):
        """
        Negative layers raises ManifestError.
        """
        data = {**VALID_MANIFEST_DATA, "layers": -1}
        path = write_manifest(tmp_path, data)

        with pytest.raises(ManifestError, match="layers"):
            load_manifest(path)

    def test_name_with_spaces(self, tmp_path):
        """
        A name containing spaces raises ManifestError.
        """
        data = {**VALID_MANIFEST_DATA, "name": "my model"}
        path = write_manifest(tmp_path, data)

        with pytest.raises(ManifestError, match="name"):
            load_manifest(path)


class TestFileErrors:
    """
    Tests for file-level errors.
    """

    def test_file_not_found(self):
        """
        A nonexistent file path raises ManifestError.
        """
        with pytest.raises(ManifestError, match="not found"):
            load_manifest("/nonexistent/path/manifest.yaml")
