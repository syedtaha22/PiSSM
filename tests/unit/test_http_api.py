"""
Tests for the orchestrator's FastAPI HTTP API.

Covers FR-NM-04 (node listing), FR-MR-01/03/04 (model registration,
validation, listing). Tests call the FastAPI app directly through
TestClient, using real NodeRegistry and ModelRegistry instances -
no gRPC or model loading involved.
"""

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
        from inference.manifest import ModelManifest

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
