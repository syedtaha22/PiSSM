"""
Tests for inference.weights: checkpoint metadata reading and file
resolution shared by the orchestrator (metadata only) and workers
(actual file fetch).
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import tqdm as tqdm_module

from inference.weights import fetch_source_file, read_checkpoint_metadata

DUMMY_CHECKPOINT = "checkpoints/dummy-mamba-tiny"


class TestLoggingSuppression:
    """
    Tests that importing this module quiets noisy third-party loggers
    on its own, without depending on inference.loader having been
    imported first - the orchestrator's checkpoint-metadata reads go
    through this module directly and never import inference.loader.
    """

    def test_httpx_logger_quieted(self):
        """
        The httpx logger (every HTTP request huggingface_hub makes) is
        set to WARNING, so per-request INFO lines don't flood the
        console.
        """
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_huggingface_hub_logger_quieted(self):
        """
        The huggingface_hub logger (e.g. the no-token rate-limit nag)
        is set to ERROR.
        """
        assert logging.getLogger("huggingface_hub").level == logging.ERROR


class TestReadCheckpointMetadata:
    """
    Tests for read_checkpoint_metadata against a local checkpoint (real,
    fast, no network) and a repo-id-shaped string (mocked).
    """

    def test_local_checkpoint_returns_real_weight_map(self):
        """
        Reading the local dummy checkpoint returns a weight_map with the
        embedding and per-layer tensor names, using the real safetensors
        header - no network involved.
        """
        metadata = read_checkpoint_metadata(DUMMY_CHECKPOINT)

        assert "backbone.embeddings.weight" in metadata.weight_map
        assert "backbone.layers.0.mixer.in_proj.weight" in metadata.weight_map

    def test_local_checkpoint_lm_head_absent_when_tied(self):
        """
        dummy-mamba-tiny ties lm_head to the input embedding (the same
        default as every mamba-*-hf checkpoint), so lm_head.weight has
        no entry of its own in the weight map.
        """
        metadata = read_checkpoint_metadata(DUMMY_CHECKPOINT)

        assert "lm_head.weight" not in metadata.weight_map

    @patch("inference.weights.os.path.isdir", return_value=False)
    @patch("inference.weights.get_safetensors_metadata")
    def test_repo_id_calls_get_safetensors_metadata(self, mock_get, mock_isdir):
        """
        A checkpoint string that isn't a local directory is treated as
        an HF repo id and read via get_safetensors_metadata, not the
        local variant.
        """
        mock_get.return_value = MagicMock()

        read_checkpoint_metadata("state-spaces/mamba-130m-hf")

        mock_get.assert_called_once_with("state-spaces/mamba-130m-hf")


class TestFetchSourceFile:
    """
    Tests for fetch_source_file: resolves a checkpoint file to a local
    path, either directly (local checkpoint) or via download (repo id).
    """

    def test_local_checkpoint_returns_path_without_network(self):
        """
        For a local checkpoint directory, the file is resolved directly
        under that directory - hf_hub_download is never touched.
        """
        with patch("inference.weights.hf_hub_download") as mock_download:
            path = fetch_source_file(DUMMY_CHECKPOINT, "model.safetensors")

        assert path == Path(DUMMY_CHECKPOINT) / "model.safetensors"
        mock_download.assert_not_called()

    @patch("inference.weights.os.path.isdir", return_value=False)
    @patch("inference.weights.hf_hub_download")
    def test_repo_id_calls_hf_hub_download(self, mock_download, mock_isdir):
        """
        For a repo id, the file is fetched via hf_hub_download with the
        checkpoint as repo_id and the tensor location's filename.
        """
        mock_download.return_value = "/fake/cache/path/model.safetensors"

        path = fetch_source_file("state-spaces/mamba-130m-hf", "model.safetensors")

        _, kwargs = mock_download.call_args
        assert kwargs["repo_id"] == "state-spaces/mamba-130m-hf"
        assert kwargs["filename"] == "model.safetensors"
        assert path == Path("/fake/cache/path/model.safetensors")

    @patch("inference.weights.os.path.isdir", return_value=False)
    @patch("inference.weights.hf_hub_download")
    def test_repo_id_download_shows_a_progress_bar(self, mock_download, mock_isdir):
        """
        The download is given a tqdm_class so hf_hub_download shows a
        real progress bar for the file transfer - this is the only
        part of a shard load that can take a while, and a worker with
        nothing on screen looks hung. The bar clears itself on
        completion (leave=False) rather than littering the terminal.
        """
        mock_download.return_value = "/fake/cache/path/model.safetensors"

        fetch_source_file("state-spaces/mamba-130m-hf", "model.safetensors")

        _, kwargs = mock_download.call_args
        tqdm_class = kwargs["tqdm_class"]
        assert issubclass(tqdm_class, tqdm_module.tqdm)
        bar = tqdm_class(total=1, disable=True)
        try:
            assert bar.leave is False
        finally:
            bar.close()
