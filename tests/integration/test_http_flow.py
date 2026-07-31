"""
Integration tests for the full inference HTTP flow.

Spins up two in-process InferenceServiceServicer-backed gRPC servers
(simulating two worker nodes) and a PipelineCallbackServicer-backed
gRPC server (simulating the orchestrator's callback endpoint), then
drives a real pipeline through the orchestrator's FastAPI app end to
end: POST /models, then POST /infer. Uses the tiny random-weight
dummy-mamba-tiny checkpoint (see scripts/make_dummy_manifest.py)
instead of a full-size model, since this only needs to exercise the
real dispatch/load/infer plumbing, not model quality.
"""

from concurrent import futures

import grpc
from fastapi.testclient import TestClient

from inference.service import InferenceServiceServicer
from orchestrator.http_api import create_app
from orchestrator.pipeline import PipelineCallbackServicer
from orchestrator.worker_client import _CHANNEL_OPTIONS
from proto.generated import inference_pb2_grpc

MANIFEST_YAML = """
name: dummy-mamba-tiny
arch: mamba
checkpoint: checkpoints/dummy-mamba-tiny
layers: 4
hidden_dim: 64
state_dim: 8
input_type: text
tokenizer: EleutherAI/gpt-neox-20b
"""


def _start_inference_worker():
    """
    Start an in-process gRPC server hosting a real InferenceServiceServicer.

    Returns
    -------
    tuple[grpc.Server, int]
        The running server and the port it bound to.
    """
    servicer = InferenceServiceServicer()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4), options=_CHANNEL_OPTIONS
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port


def _start_callback_server(result_store):
    """
    Start an in-process gRPC server hosting PipelineCallbackServicer.

    Parameters
    ----------
    result_store : ResultStore
        The store the servicer delivers results into.

    Returns
    -------
    tuple[grpc.Server, int]
        The running server and the port it bound to.
    """
    servicer = PipelineCallbackServicer(result_store)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4), options=_CHANNEL_OPTIONS
    )
    inference_pb2_grpc.add_PipelineCallbackServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port


class TestHttpInferenceFlow:
    """
    End-to-end test of POST /models -> POST /infer over a real
    two-node pipeline.
    """

    def test_infer_over_two_node_pipeline(self, registry, model_registry, result_store):
        """
        Registering two workers and running inference through the HTTP
        API produces non-empty output with per-node timing for both.
        """
        worker_servers = []
        callback_server = None
        try:
            for i in range(2):
                server, port = _start_inference_worker()
                worker_servers.append(server)
                registry.update_node(
                    node_id=f"node-{i}",
                    ip_address="localhost",
                    available_ram_mb=3800,
                    total_ram_mb=4096,
                    cpu_count=4,
                    arch="aarch64",
                    os_name="Linux",
                    os_version="test",
                    inference_port=port,
                )

            callback_server, callback_port = _start_callback_server(result_store)

            app = create_app(
                registry,
                model_registry,
                result_store,
                callback_address=f"localhost:{callback_port}",
            )
            client = TestClient(app)

            models_resp = client.post("/models", json={"manifest_yaml": MANIFEST_YAML})
            assert models_resp.status_code == 201

            infer_resp = client.post(
                "/infer",
                json={
                    "model_name": "dummy-mamba-tiny",
                    "input": "Hey how are you doing?",
                    "max_new_tokens": 2,
                },
            )

            assert infer_resp.status_code == 200
            body = infer_resp.json()
            assert isinstance(body["output"], str)
            assert len(body["output"]) > 0
            assert body["num_nodes"] == 2
            assert len(body["node_latencies_ms"]) == 2
        finally:
            for server in worker_servers:
                server.stop(grace=2)
            if callback_server is not None:
                callback_server.stop(grace=2)
