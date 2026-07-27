"""
Tests for the orchestrator's FastAPI HTTP API.

Covers FR-NM-04 (node listing), FR-MR-01/03/04 (model registration,
validation, listing), and FR-IE-01/04/05/06 (POST /infer). Tests call
the FastAPI app directly through TestClient, using real NodeRegistry
and ModelRegistry instances. POST /infer tests mock plan_dispatch,
ModelStore, PipelineRunner, and AutoTokenizer - no gRPC, model
download, or real inference involved.
"""

from unittest.mock import MagicMock, patch

import torch

from inference.manifest import ModelManifest
from orchestrator.dispatch import DispatchError

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


class TestGetNodes:
    """
    Tests for GET /nodes.
    """

    def test_empty_registry_returns_empty_list(self, fastapi_test_client):
        """
        An empty node registry returns an empty list.
        """
        response = fastapi_test_client.get("/nodes")

        assert response.status_code == 200
        assert response.json() == []

    def test_populated_registry_returns_nodes(self, fastapi_test_client, registry):
        """
        A registered node is returned with its fields.
        """
        registry.update_node(
            node_id="node-1",
            ip_address="192.168.1.10",
            available_ram_mb=3800,
            total_ram_mb=4096,
            cpu_count=4,
            arch="aarch64",
            os_name="Linux",
            os_version="6.6.31+rpt-rpi-2712",
            inference_port=50052,
        )

        response = fastapi_test_client.get("/nodes")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["node_id"] == "node-1"
        assert body[0]["status"] == "available"
        assert body[0]["inference_port"] == 50052


class TestGetModels:
    """
    Tests for GET /models.
    """

    def test_empty_registry_returns_empty_list(self, fastapi_test_client):
        """
        An empty model registry returns an empty list.
        """
        response = fastapi_test_client.get("/models")

        assert response.status_code == 200
        assert response.json() == []

    def test_populated_registry_returns_models(
        self, fastapi_test_client, model_registry
    ):
        """
        A registered model is returned with its fields.
        """
        model_registry.register(
            ModelManifest(
                name="mamba-130m",
                arch="mamba",
                checkpoint="state-spaces/mamba-130m-hf",
                layers=24,
                hidden_dim=768,
                state_dim=16,
                input_type="text",
                tokenizer="EleutherAI/gpt-neox-20b",
            )
        )

        response = fastapi_test_client.get("/models")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["name"] == "mamba-130m"
        assert body[0]["arch"] == "mamba"


class TestPostModels:
    """
    Tests for POST /models.
    """

    def test_valid_manifest_registers_and_is_retrievable(self, fastapi_test_client):
        """
        A valid manifest is registered and appears in a subsequent listing.
        """
        response = fastapi_test_client.post(
            "/models", json={"manifest_yaml": VALID_MANIFEST_YAML}
        )

        assert response.status_code == 201
        assert response.json()["name"] == "mamba-130m"

        listing = fastapi_test_client.get("/models")
        assert len(listing.json()) == 1

    def test_missing_field_returns_400(self, fastapi_test_client):
        """
        A manifest missing a required field returns 400 naming the field.
        """
        bad_yaml = "name: test-model\narch: mamba\n"

        response = fastapi_test_client.post("/models", json={"manifest_yaml": bad_yaml})

        assert response.status_code == 400
        assert "checkpoint" in response.json()["detail"]

    def test_unsupported_arch_returns_400(self, fastapi_test_client):
        """
        An unsupported architecture returns 400 naming the field.
        """
        bad_yaml = VALID_MANIFEST_YAML.replace("arch: mamba", "arch: rnn")

        response = fastapi_test_client.post("/models", json={"manifest_yaml": bad_yaml})

        assert response.status_code == 400
        assert "arch" in response.json()["detail"]

    def test_duplicate_name_returns_409(self, fastapi_test_client):
        """
        Registering the same model name twice returns 409 on the second call.
        """
        fastapi_test_client.post("/models", json={"manifest_yaml": VALID_MANIFEST_YAML})

        response = fastapi_test_client.post(
            "/models", json={"manifest_yaml": VALID_MANIFEST_YAML}
        )

        assert response.status_code == 409

    def test_invalid_yaml_returns_400(self, fastapi_test_client):
        """
        Malformed YAML returns 400 rather than a server error.
        """
        response = fastapi_test_client.post(
            "/models", json={"manifest_yaml": "not: valid: yaml: [structure"}
        )

        assert response.status_code == 400

    def test_non_mapping_yaml_returns_400(self, fastapi_test_client):
        """
        YAML that parses but is not a mapping (e.g. a list) returns 400.
        """
        response = fastapi_test_client.post(
            "/models", json={"manifest_yaml": "- one\n- two\n"}
        )

        assert response.status_code == 400


def _registered_manifest(model_registry):
    """
    Register and return a valid mamba-130m manifest in model_registry.

    Parameters
    ----------
    model_registry : ModelRegistry
        The registry to register the manifest into.
    """
    manifest = ModelManifest(
        name="mamba-130m",
        arch="mamba",
        checkpoint="state-spaces/mamba-130m-hf",
        layers=24,
        hidden_dim=768,
        state_dim=16,
        input_type="text",
        tokenizer="EleutherAI/gpt-neox-20b",
    )
    model_registry.register(manifest)
    return manifest


def _fake_pipeline_session():
    """
    Build mocks for a one-node dispatch plan, tokenizer, and pipeline result.

    Returns
    -------
    tuple[MagicMock, MagicMock, MagicMock]
        (fake_plan, fake_tokenizer, fake_runner) ready to patch in as
        plan_dispatch's return value, AutoTokenizer.from_pretrained's
        return value, and PipelineRunner's return value.
    """
    fake_plan = MagicMock()
    fake_plan.assignments = [MagicMock()]

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])
    fake_tokenizer.decode.return_value = "hello world"

    fake_result = MagicMock()
    fake_result.output_tensor = torch.zeros(1, 3, 10)
    fake_result.node_latencies_ms = [5.0]
    fake_result.node_peak_memory_mb = [260]

    fake_runner = MagicMock()
    fake_runner.run_forward.return_value = fake_result

    return fake_plan, fake_tokenizer, fake_runner


class TestPostInfer:
    """
    Tests for POST /infer.
    """

    def test_unknown_model_returns_404(self, fastapi_test_client):
        """
        Requesting inference for an unregistered model returns 404.
        """
        response = fastapi_test_client.post(
            "/infer", json={"model_name": "nonexistent", "input": "hi"}
        )

        assert response.status_code == 404

    def test_no_available_nodes_returns_503(self, fastapi_test_client, model_registry):
        """
        No available nodes in the registry returns 503, not a 500.
        """
        _registered_manifest(model_registry)

        with patch(
            "orchestrator.http_api.plan_dispatch",
            side_effect=DispatchError("no available nodes in the registry"),
        ):
            response = fastapi_test_client.post(
                "/infer", json={"model_name": "mamba-130m", "input": "hi"}
            )

        assert response.status_code == 503
        assert "no available" in response.json()["detail"]

    def test_success_path_returns_output_and_latency(
        self, fastapi_test_client, model_registry
    ):
        """
        A successful inference returns output text and timing/memory fields.
        """
        _registered_manifest(model_registry)
        fake_plan, fake_tokenizer, fake_runner = _fake_pipeline_session()

        with (
            patch("orchestrator.http_api.plan_dispatch", return_value=fake_plan),
            patch("orchestrator.http_api.ModelStore"),
            patch("orchestrator.http_api.PipelineRunner", return_value=fake_runner),
            patch(
                "orchestrator.http_api.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            response = fastapi_test_client.post(
                "/infer",
                json={"model_name": "mamba-130m", "input": "hi", "max_new_tokens": 1},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["output"] == "hello world"
        assert body["node_latencies_ms"] == [5.0]
        assert body["peak_memory_mb"] == [260]
        assert body["num_nodes"] == 1
        assert body["latency_ms"] >= 0
        fake_runner.load.assert_called_once()

    def test_second_call_reuses_resident_runner(
        self, fastapi_test_client, model_registry
    ):
        """
        A second request for the same model reuses the loaded runner
        rather than dispatching and loading shards again.
        """
        _registered_manifest(model_registry)
        fake_plan, fake_tokenizer, fake_runner = _fake_pipeline_session()

        with (
            patch(
                "orchestrator.http_api.plan_dispatch", return_value=fake_plan
            ) as mock_plan_dispatch,
            patch("orchestrator.http_api.ModelStore"),
            patch(
                "orchestrator.http_api.PipelineRunner", return_value=fake_runner
            ) as mock_runner_cls,
            patch(
                "orchestrator.http_api.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            fastapi_test_client.post(
                "/infer",
                json={"model_name": "mamba-130m", "input": "hi", "max_new_tokens": 1},
            )
            fastapi_test_client.post(
                "/infer",
                json={
                    "model_name": "mamba-130m",
                    "input": "hi again",
                    "max_new_tokens": 1,
                },
            )

        assert mock_runner_cls.call_count == 1
        assert mock_plan_dispatch.call_count == 1
        assert fake_runner.load.call_count == 1
        assert fake_runner.run_forward.call_count == 2
