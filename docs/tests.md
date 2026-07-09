# Test Cases

This document catalogs every test case in the PiSSM project. Each test is linked to a specific functional or non-functional requirement from the SRS. The requirement reference sections below provide a quick lookup so test tables can be read without switching to the SRS.

## Functional Requirements Reference

### Node Management

- **FR-NM-01:** Each worker node shall broadcast a heartbeat to the orchestrator at a configurable interval (default: 2 seconds).
- **FR-NM-02:** The orchestrator shall maintain a live node registry tracking each node's identity, available RAM, and last heartbeat timestamp.
- **FR-NM-03:** A node that misses three consecutive heartbeats shall be marked as unavailable and excluded from dispatch decisions.
- **FR-NM-04:** The orchestrator shall support listing all nodes with their current status.
- **FR-NM-05:** Nodes shall automatically rejoin the cluster upon restart without manual intervention.

### Model Registry

- **FR-MR-01:** The system shall maintain a registry of submitted models, storing the manifest, checkpoint path, and compilation status.
- **FR-MR-02:** A user shall be able to submit a model via `compile <manifest.yaml>` (TUI) or the WebUI upload form.
- **FR-MR-03:** The system shall validate the manifest on submission, reporting clear errors for missing fields or unsupported architecture values.
- **FR-MR-04:** A user shall be able to list registered models via `ls models` (TUI) or the WebUI model list.
- **FR-MR-05:** A user shall be able to delete a registered model.

### Dispatch Engine

- **FR-DE-01:** On receiving a run request, the orchestrator shall query available RAM across all live nodes.
- **FR-DE-02:** The dispatch engine shall apply dispatch rules in order: single node, layer parallel, quantized fallback, rejection.
- **FR-DE-03:** The dispatch decision and resulting topology shall be recorded and viewable by the user.
- **FR-DE-04:** The topology shall be inspectable and manually overridable via `vim topology.yaml` (TUI) or the topology editor (WebUI).

### Inference Execution

- **FR-IE-01:** The system shall accept user input appropriate to the model's input_type (text prompt, numeric array, or audio file path).
- **FR-IE-02:** For text input, the system shall run the associated tokenizer before passing data to the first worker shard.
- **FR-IE-03:** For timeseries or audio input, the system shall normalize and chunk the input to the model's configured sequence length.
- **FR-IE-04:** Pipeline execution shall pass activations between worker nodes via gRPC.
- **FR-IE-05:** The final node shall return the output to the orchestrator, which returns it to the user interface.
- **FR-IE-06:** Inference latency (end-to-end) and per-node execution time shall be recorded for every request.

### Fault Tolerance

- **FR-FT-01:** If a worker node becomes unavailable during an active inference request, the orchestrator shall detect the failure and abort the current request with an informative error.
- **FR-FT-02:** The orchestrator shall immediately attempt to re-dispatch the same request to remaining live nodes using the dispatch rules.
- **FR-FT-03:** The node registry shall be updated immediately upon failure detection.
- **FR-FT-04:** Recovery of a failed node shall be automatic upon reconnection.

### TUI

- **FR-TUI-01:** The TUI shall launch as a standalone terminal application and connect to the orchestrator's HTTP API.
- **FR-TUI-02:** The TUI shall support commands: `listn`, `ls models`, `compile`, `run`, `vim topology.yaml`, `status`, `logs`.
- **FR-TUI-03:** The TUI shall display a persistent status bar showing live node count and cluster RAM utilization.
- **FR-TUI-04:** The TUI shall display inference output and timing results inline after a `run` command completes.

### WebUI

- **FR-WUI-01:** The WebUI shall be served by the orchestrator and accessible at `http://<orchestrator-ip>:8080`.
- **FR-WUI-02:** The WebUI shall provide a dashboard showing node status, cluster RAM, and recent inference history.
- **FR-WUI-03:** The WebUI shall allow model submission via a file upload form accepting a manifest YAML and checkpoint file.
- **FR-WUI-04:** The WebUI shall provide an inference panel where the user selects a registered model, enters input, and views output.
- **FR-WUI-05:** The WebUI shall display the active topology visually.
- **FR-WUI-06:** The WebUI shall display per-request benchmark data (latency, throughput, memory usage per node).

### Benchmarking

- **FR-BM-01:** The system shall record for every inference request: total latency, per-node execution time, activation transfer time, and peak memory usage per node.
- **FR-BM-02:** The system shall expose a benchmark mode where the same input is run N times and aggregate statistics are reported.
- **FR-BM-03:** Benchmark results shall be exportable as CSV.

## Non-Functional Requirements Reference

- **NFR-01 (Performance):** End-to-end inference latency for Mamba-130M on a single node shall be under 5 seconds for a 512-token input at steady state.
- **NFR-02 (Reliability):** The system shall handle single-node failure without crashing the orchestrator or other workers.
- **NFR-03 (Portability):** All software shall run on Raspberry Pi OS (64-bit, Debian Bookworm base) without modification.
- **NFR-04 (Usability):** A user unfamiliar with the system shall be able to submit and run a model within 10 minutes using the WebUI, given a valid manifest and checkpoint.
- **NFR-05 (Observability):** All inter-node gRPC calls shall be logged with timestamps.

---

## Unit Tests

### NodeRegistry (`tests/unit/test_node_registry.py`)

Component under test: `orchestrator.node_registry.NodeRegistry` - a thread-safe in-memory store of worker node state. All unit tests use an injectable mock clock to control time deterministically.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-NR-01 | FR-NM-02 | TestNodeRegistration | Register new node | Verify that registering a node stores all fields correctly and sets the initial status to available. | Empty registry with mock clock at t=100.0. | node_id="node-1", ip="192.168.1.10", available_ram=3800, total_ram=4096, cpu_count=4, arch="aarch64", os_name="Linux", os_version="6.6.31+rpt-rpi-2712", inference_port=50052. | 1. Create registry with mock clock at t=100.0. 2. Call `update_node` with test data. 3. Call `get_node("node-1")`. | All fields match input, status="available", last_heartbeat=100.0, first_seen=100.0, inference_port=50052. |
| TC-NR-02 | FR-NM-02 | TestNodeRegistration | Update existing node | Verify that a second heartbeat updates RAM and timestamp but preserves first_seen. | Registry with mock clock at t=100.0, one registered node. | Same node_id, first call available_ram=3800, second call available_ram=3500 after 2s advance. | 1. Register node-1 with available_ram=3800 at t=100.0. 2. Advance clock by 2.0s. 3. Register node-1 again with available_ram=3500. 4. Call `get_node("node-1")`. | available_ram_mb=3500, last_heartbeat=102.0, first_seen=100.0. |
| TC-NR-03 | FR-NM-02 | TestNodeRegistration | Lookup unknown node | Verify that looking up an unregistered node returns None. | Empty registry. | node_id="nonexistent". | 1. Create empty registry. 2. Call `get_node("nonexistent")`. | Returns None. |
| TC-NL-01 | FR-NM-04 | TestNodeListing | List nodes on empty registry | Verify that listing on an empty registry returns an empty list. | Empty registry. | None. | 1. Create empty registry. 2. Call `list_nodes()`. | Returns []. |
| TC-NL-02 | FR-NM-04 | TestNodeListing | List all registered nodes | Verify that all registered nodes are returned with no filter. | Empty registry. | Three nodes: node-1, node-2, node-3. | 1. Register node-1, node-2, node-3. 2. Call `list_nodes()`. | Returns 3 items with node_ids {"node-1", "node-2", "node-3"}. |
| TC-NL-03 | FR-NM-04 | TestNodeListing | Filter nodes by status | Verify that filtering by status returns only matching nodes after one is reaped. | Registry with heartbeat_interval=1.0s, missed_threshold=3, mock clock at t=0.0. | Two nodes: node-1, node-2. | 1. Register both at t=0. 2. Advance clock by 4.0s. 3. Register node-1 again. 4. Call `reap_stale_nodes()`. 5. Call `list_nodes` with each filter. | Available list: [node-1]. Unavailable list: [node-2]. |
| TC-FD-01 | FR-NM-03 | TestFailureDetection | Reap stale node | Verify that a node past the timeout is marked unavailable and returned by the reaper. | Registry with heartbeat_interval=2.0s, missed_threshold=3 (timeout=6.0s), mock clock at t=0.0. | One node: node-1. | 1. Register node-1 at t=0. 2. Advance clock by 7.0s. 3. Call `reap_stale_nodes()`. 4. Call `get_node("node-1")`. | Reaper returns ["node-1"]. Node status is "unavailable". |
| TC-FD-02 | FR-NM-03 | TestFailureDetection | Do not reap fresh nodes | Verify that nodes within the timeout window are not reaped. | Registry with heartbeat_interval=2.0s, missed_threshold=3 (timeout=6.0s), mock clock at t=0.0. | One node: node-1. | 1. Register node-1 at t=0. 2. Advance clock by 1.0s. 3. Call `reap_stale_nodes()`. 4. Call `get_node("node-1")`. | Reaper returns []. Node status is "available". |
| TC-FD-03 | FR-NM-03 | TestFailureDetection | Do not reap already unavailable | Verify that a second reap does not report the same node again. | Registry with heartbeat_interval=2.0s, missed_threshold=3 (timeout=6.0s), mock clock at t=0.0. | One node: node-1. | 1. Register node-1 at t=0. 2. Advance clock by 7.0s. 3. Call `reap_stale_nodes()`. 4. Call `reap_stale_nodes()` again. | First reap returns ["node-1"]. Second reap returns []. |
| TC-FD-04 | FR-NM-03 | TestFailureDetection | Timeout calculation | Verify that timeout_s equals heartbeat_interval_s * missed_threshold. | None. | heartbeat_interval_s=2.0, missed_threshold=3. | 1. Create registry with given parameters. 2. Read `timeout_s`. | timeout_s == 6.0. |
| TC-AR-01 | FR-NM-05 | TestAutoRejoin | Rejoin after unavailable | Verify that a previously unavailable node is restored to available on a new heartbeat. | Registry with heartbeat_interval=2.0s, missed_threshold=3 (timeout=6.0s), mock clock at t=0.0. | One node: node-1. | 1. Register at t=0. 2. Advance by 7.0s. 3. Reap. 4. Advance by 1.0s. 5. Register again. 6. Call `get_node("node-1")`. | status="available", last_heartbeat=8.0, first_seen=0.0. |
| TC-TS-01 | NFR-02 | TestThreadSafety | Concurrent updates | Verify that 10 threads updating different nodes simultaneously causes no corruption. | Empty registry with default clock. | 10 threads, each updating a unique node_id 100 times. | 1. Create empty registry. 2. Spawn 10 threads. 3. Join all. 4. Check for exceptions. 5. Call `list_nodes()`. | No exceptions. Registry contains exactly 10 nodes. |

### System Info (`tests/unit/test_system_info.py`)

Component under test: `worker.system_info` - functions that gather node identity and hardware state for heartbeat payloads. These wrap `socket`, `platform`, and `psutil` calls.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-SI-01 | FR-NM-01 | TestNodeIdentity | Node ID is non-empty string | Verify that get_node_id returns a non-empty string. | None. | None. | 1. Call `get_node_id()`. | Returns a non-empty string. |
| TC-SI-02 | FR-NM-01 | TestNodeIdentity | IP address is valid IPv4 | Verify that get_ip_address returns a dotted-quad IPv4 string. | Network connectivity. | None. | 1. Call `get_ip_address()`. 2. Split by ".". 3. Validate each octet. | Four octets, each 0-255. |
| TC-SI-03 | FR-NM-02 | TestMemoryInfo | Available RAM is positive int | Verify that get_available_ram_mb returns a positive integer. | None. | None. | 1. Call `get_available_ram_mb()`. | Returns int > 0. |
| TC-SI-04 | FR-NM-02 | TestMemoryInfo | Total RAM is positive int | Verify that get_total_ram_mb returns a positive integer. | None. | None. | 1. Call `get_total_ram_mb()`. | Returns int > 0. |
| TC-SI-05 | FR-NM-02 | TestMemoryInfo | Available RAM does not exceed total | Verify that available RAM is less than or equal to total RAM. | None. | None. | 1. Call both functions. 2. Compare. | available <= total. |
| TC-SI-06 | FR-NM-02 | TestHardwareInfo | CPU count is positive int | Verify that get_cpu_count returns a positive integer. | None. | None. | 1. Call `get_cpu_count()`. | Returns int > 0. |
| TC-SI-07 | FR-NM-02 | TestHardwareInfo | Architecture is non-empty string | Verify that get_arch returns a non-empty string. | None. | None. | 1. Call `get_arch()`. | Returns non-empty string. |
| TC-SI-08 | FR-NM-02 | TestOSInfo | OS name is non-empty string | Verify that get_os_name returns a non-empty string. | None. | None. | 1. Call `get_os_name()`. | Returns non-empty string. |
| TC-SI-09 | FR-NM-02 | TestOSInfo | OS version is non-empty string | Verify that get_os_version returns a non-empty string. | None. | None. | 1. Call `get_os_version()`. | Returns non-empty string. |

### NodeServiceServicer (`tests/unit/test_heartbeat_service.py`)

Component under test: `orchestrator.service.NodeServiceServicer` - gRPC handler that bridges heartbeat and status requests to the NodeRegistry. Tests call servicer methods directly with mock context objects, no network I/O.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-HS-01 | FR-NM-01 | TestHeartbeatRPC | Heartbeat returns acknowledged | Verify that a valid heartbeat returns acknowledged=True. | Empty registry. | Default heartbeat request. | 1. Create servicer with empty registry. 2. Call `Heartbeat` with request. | response.acknowledged is True. |
| TC-HS-02 | FR-NM-02 | TestHeartbeatRPC | Heartbeat registers node | Verify that after a heartbeat the node exists in the registry with correct fields including inference_port. | Empty registry. | node_id="node-1", ip="192.168.1.10", available_ram=3800, total_ram=4096, cpu_count=4, arch="aarch64", inference_port=50052. | 1. Call `Heartbeat`. 2. Call `registry.get_node("node-1")`. | Node exists with all fields matching, status="available", inference_port=50052. |
| TC-HS-03 | FR-NM-01 | TestHeartbeatRPC | Heartbeat returns configured interval | Verify that the response carries the heartbeat interval configured on the servicer. | Empty registry, servicer with heartbeat_interval_ms=5000. | Default heartbeat request. | 1. Create servicer with interval=5000. 2. Call `Heartbeat`. | response.heartbeat_interval_ms == 5000. |
| TC-HS-04 | FR-NM-01 | TestHeartbeatRPC | Heartbeat default interval | Verify that the default heartbeat interval is 2000ms. | Empty registry. | Default heartbeat request. | 1. Create servicer with default config. 2. Call `Heartbeat`. | response.heartbeat_interval_ms == 2000. |
| TC-HS-05 | FR-NM-04 | TestReportStatusRPC | ReportStatus returns correct fields | Verify that ReportStatus returns all node fields for a known node. | Registry with one node registered via Heartbeat. | node_id="node-1". | 1. Send Heartbeat. 2. Call `ReportStatus(node_id="node-1")`. | All fields match, status=AVAILABLE. |
| TC-HS-06 | FR-NM-04 | TestReportStatusRPC | ReportStatus unknown node | Verify that ReportStatus sets NOT_FOUND for an unknown node. | Empty registry. | node_id="nonexistent". | 1. Call `ReportStatus(node_id="nonexistent")`. | context.set_code called with NOT_FOUND, context.set_details called. |

### HeartbeatClient (`tests/unit/test_heartbeat_client.py`)

Component under test: `worker.heartbeat.HeartbeatClient` - periodically sends gRPC heartbeats to the orchestrator from a background thread. Each test creates its own in-process gRPC server with a matching heartbeat interval so the server does not override the client's test interval.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-HC-01 | FR-NM-01 | TestHeartbeatClientLifecycle | Client sends heartbeat | Verify the client sends at least one heartbeat after starting. | In-process gRPC server with interval=200ms. | node_id="test-node", interval=0.2s, inference_port=50052. | 1. Start client. 2. Wait 0.5s. 3. Stop client. 4. Check registry. | Node exists with status="available". |
| TC-HC-02 | FR-NM-01 | TestHeartbeatClientLifecycle | Client stops cleanly | Verify that after stop(), is_running is False. | In-process gRPC server with interval=200ms. | node_id="test-node", interval=0.2s, inference_port=50052. | 1. Start client. 2. Assert is_running=True. 3. Stop client. | is_running is False. |
| TC-HC-03 | FR-NM-01 | TestHeartbeatClientLifecycle | Client sends multiple heartbeats | Verify heartbeats continue periodically, not just on startup. | In-process gRPC server with interval=200ms. | node_id="test-node", interval=0.2s, inference_port=50052. | 1. Start client. 2. Wait 1.0s. 3. Read last_heartbeat. 4. Compare to current time. | last_heartbeat is within 0.5s of current time. |
| TC-HC-04 | NFR-02 | TestHeartbeatClientResilience | Client handles server unavailable | Verify the client does not crash when the orchestrator is unreachable. | No server running on target port. | orchestrator_address="localhost:1", interval=0.2s. | 1. Start client. 2. Wait 0.5s. 3. Stop client. | is_running is False, no exceptions. |

### Model Manifest (`tests/unit/test_manifest.py`)

Component under test: `inference.manifest` - YAML manifest parsing and validation. No torch dependency. Uses pytest `tmp_path` for temporary YAML files.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-MF-01 | FR-MR-01 | TestLoadValidManifest | Load valid manifest | Verify a valid YAML produces a ModelManifest with all fields. | None. | Full valid manifest data. | 1. Write YAML to tmp file. 2. Call `load_manifest()`. | All 8 fields match input. |
| TC-MF-02 | FR-MR-01 | TestLoadValidManifest | Manifest is frozen | Verify ModelManifest is immutable. | None. | Valid manifest. | 1. Load manifest. 2. Try to assign a field. | Raises AttributeError. |
| TC-MF-03 | FR-MR-03 | TestMissingFields | Missing required field (parametrized x8) | Verify each missing required field raises ManifestError. | None. | Manifest with one field removed. | 1. Remove field. 2. Call `load_manifest()`. | Raises ManifestError matching field name. |
| TC-MF-04 | FR-MR-03 | TestInvalidValues | Unsupported arch | Verify unsupported architecture raises ManifestError. | None. | arch="rnn". | 1. Call `load_manifest()`. | Raises ManifestError matching "arch". |
| TC-MF-05 | FR-MR-03 | TestInvalidValues | Unsupported input type | Verify unsupported input type raises ManifestError. | None. | input_type="video". | 1. Call `load_manifest()`. | Raises ManifestError matching "input_type". |
| TC-MF-06 | FR-MR-03 | TestInvalidValues | Layers zero | Verify zero layers raises ManifestError. | None. | layers=0. | 1. Call `load_manifest()`. | Raises ManifestError matching "layers". |
| TC-MF-07 | FR-MR-03 | TestInvalidValues | Layers negative | Verify negative layers raises ManifestError. | None. | layers=-1. | 1. Call `load_manifest()`. | Raises ManifestError matching "layers". |
| TC-MF-08 | FR-MR-03 | TestInvalidValues | Name with spaces | Verify name with spaces raises ManifestError. | None. | name="my model". | 1. Call `load_manifest()`. | Raises ManifestError matching "name". |
| TC-MF-09 | FR-MR-03 | TestFileErrors | File not found | Verify nonexistent path raises ManifestError. | None. | path="/nonexistent/manifest.yaml". | 1. Call `load_manifest()`. | Raises ManifestError matching "not found". |

### Model Registry (`tests/unit/test_model_registry.py`)

Component under test: `inference.model_registry.ModelRegistry` - thread-safe in-memory store of validated model manifests. No torch dependency.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-MR-01 | FR-MR-01 | TestRegisterAndGet | Register and get | Verify a registered manifest can be retrieved by name. | Empty registry. | Valid manifest. | 1. Register manifest. 2. Call `get(name)`. | Returns manifest with matching fields. |
| TC-MR-02 | FR-MR-01 | TestRegisterAndGet | Duplicate raises | Verify registering the same name twice raises ValueError. | Registry with one model. | Same manifest. | 1. Register. 2. Register again. | Raises ValueError matching model name. |
| TC-MR-03 | FR-MR-01 | TestRegisterAndGet | Get unknown | Verify looking up an unregistered model returns None. | Empty registry. | name="nonexistent". | 1. Call `get("nonexistent")`. | Returns None. |
| TC-MR-04 | FR-MR-04 | TestListModels | List empty | Verify empty registry returns empty list. | Empty registry. | None. | 1. Call `list_models()`. | Returns []. |
| TC-MR-05 | FR-MR-04 | TestListModels | List all | Verify all registered models are returned. | Empty registry. | 3 models. | 1. Register 3 models. 2. Call `list_models()`. | Returns 3 items with correct names. |
| TC-MR-06 | FR-MR-05 | TestDeleteModel | Delete existing | Verify deleting an existing model returns True and removes it. | Registry with one model. | name="mamba-130m". | 1. Delete. 2. Call `get(name)`. | Returns True, get returns None. |
| TC-MR-07 | FR-MR-05 | TestDeleteModel | Delete nonexistent | Verify deleting a nonexistent model returns False. | Empty registry. | name="nonexistent". | 1. Call `delete("nonexistent")`. | Returns False. |
| TC-MR-08 | NFR-02 | TestThreadSafety | Concurrent register | Verify 20 threads registering different models causes no corruption. | Empty registry. | 20 unique model names. | 1. Spawn 20 threads. 2. Join. 3. Call `list_models()`. | No exceptions, 20 models in registry. |

### Tensor Serialization (`tests/unit/test_tensor_utils.py`)

Component under test: `inference.tensor_utils` - serializes PyTorch tensors to raw bytes with shape and dtype metadata for gRPC transport, and reconstructs them on the receiving side. Requires torch, no model download.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-TU-01 | FR-IE-04 | TestRoundtrip | Roundtrip float32 | Verify a float32 tensor survives serialize-deserialize with correct values. | None. | [1.0, 2.5, -3.7] float32. | 1. Serialize. 2. Deserialize. 3. Compare. | Values match, dtype is float32. |
| TC-TU-02 | FR-IE-04 | TestRoundtrip | Roundtrip int64 | Verify an int64 tensor (token IDs) survives roundtrip. | None. | [101, 2023, 3045, 0] int64. | 1. Serialize. 2. Deserialize. 3. Compare. | Values match, dtype is int64. |
| TC-TU-03 | FR-IE-04 | TestRoundtrip | Roundtrip 2D | Verify a 2D tensor shape (1, 128) is preserved. | None. | Random int64 (1, 128). | 1. Serialize. 2. Deserialize. 3. Check shape. | Shape is (1, 128), values match. |
| TC-TU-04 | FR-IE-04 | TestRoundtrip | Roundtrip 3D | Verify a 3D tensor shape (1, 128, 768) is preserved. | None. | Random float32 (1, 128, 768). | 1. Serialize. 2. Deserialize. 3. Check shape. | Shape is (1, 128, 768), values match. |
| TC-TU-05 | FR-IE-04 | TestErrors | Unknown dtype raises | Verify unsupported dtype string raises ValueError. | None. | dtype_str="torch.bfloat16". | 1. Call `deserialize_tensor` with unsupported dtype. | Raises ValueError matching "dtype". |
| TC-TU-06 | FR-IE-04 | TestErrors | Shape mismatch raises | Verify mismatched data size and shape raises ValueError. | None. | 3-element float32 data, shape=[10]. | 1. Serialize 3 floats. 2. Deserialize with shape [10]. | Raises ValueError. |

### Model Loader (`tests/unit/test_loader.py`)

Component under test: `inference.loader` - loads HuggingFace models and tokenizers, runs tokenization and generation. Unit tests mock the model class via `patch.dict(_ARCH_TO_MODEL_CLASS)` to avoid downloading models.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-ML-01 | FR-IE-01 | TestLoadModel | Unsupported arch raises | Verify loading an unsupported architecture raises NotImplementedError. | None. | arch="s4". | 1. Call `load_model` with arch="s4". | Raises NotImplementedError matching "s4". |
| TC-ML-02 | FR-IE-01 | TestLoadModel | Load returns handle | Verify loading a model returns a ModelHandle with all fields set. | Mocked model class and tokenizer. | Default manifest. | 1. Call `load_model`. 2. Check handle fields. | name, model, tokenizer, manifest, loaded_at all set. |
| TC-ML-03 | FR-IE-01 | TestLoadModel | Load sets eval mode | Verify the model is set to eval mode after loading. | Mocked model class. | Default manifest. | 1. Call `load_model`. 2. Check mock. | `model.eval()` called once. |
| TC-ML-04 | FR-IE-01 | TestLoadModel | Load sets CPU | Verify the model is moved to CPU after loading. | Mocked model class. | Default manifest. | 1. Call `load_model`. 2. Check mock. | `model.to("cpu")` called once. |
| TC-ML-05 | FR-IE-01 | TestUnloadModel | Unload clears references | Verify unloading sets model and tokenizer to None. | ModelHandle with mock model/tokenizer. | None. | 1. Call `unload_model`. 2. Check handle. | model is None, tokenizer is None. |

### InferenceServiceServicer (`tests/unit/test_inference_service.py`)

Component under test: `inference.service.InferenceServiceServicer` - gRPC handler that bridges LoadShard/RunShard/UnloadShard requests to the model loader. Unit tests mock the loader via `patch("inference.service.load_model")` to avoid model downloads.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-IS-01 | FR-IE-01 | TestLoadShard | Load shard success | Verify LoadShard with valid parameters returns success. | Mocked loader. | Default load request. | 1. Call `LoadShard`. | success=True, memory_used_mb=260, layers_loaded=24. |
| TC-IS-02 | FR-IE-01 | TestLoadShard | Load shard stores model | Verify the model is stored in the servicer's internal dict. | Mocked loader. | Default load request. | 1. Call `LoadShard`. 2. Check `_models`. | Model present in dict. |
| TC-IS-03 | FR-IE-01 | TestLoadShard | Load shard duplicate rejects | Verify loading the same name twice returns failure. | Mocked loader, one model loaded. | Same model name twice. | 1. Load. 2. Load again. | success=False, "already loaded". |
| TC-IS-04 | FR-IE-01 | TestLoadShard | Load shard failure | Verify loader exceptions produce success=False with error message. | Mocked loader raises. | arch="s4". | 1. Call `LoadShard`. | success=False, error contains "s4". |
| TC-IS-05 | FR-IE-04 | TestRunShard | Run shard model not loaded | Verify RunShard for unloaded model sets NOT_FOUND. | Empty servicer. | model_name="mamba-130m". | 1. Call `RunShard`. | success=False, context.set_code(NOT_FOUND). |
| TC-IS-06 | FR-IE-04 | TestRunShard | Run shard generate mode | Verify RunShard in generate mode returns output tensor. | Mocked loader, model loaded, mock model.generate. | generate_mode=True. | 1. Load. 2. RunShard. | success=True, output_tensor non-empty, latency > 0. |
| TC-IS-07 | FR-IE-04 | TestRunShard | Run shard forward pass | Verify RunShard in forward-pass mode returns logits. | Mocked loader, model loaded, mock model(). | generate_mode=False. | 1. Load. 2. RunShard. | success=True, output_tensor non-empty. |
| TC-IS-08 | FR-IE-06 | TestRunShard | Run shard records latency | Verify RunShard records positive latency in response. | Mocked loader, model loaded. | Default run request. | 1. Load. 2. RunShard. | latency_ms > 0. |
| TC-IS-09 | FR-IE-01 | TestUnloadShard | Unload shard success | Verify UnloadShard removes model and returns memory freed. | Mocked loader, one model loaded. | model_name="mamba-130m". | 1. Load. 2. Unload. | success=True, model removed from dict. |
| TC-IS-10 | FR-IE-01 | TestUnloadShard | Unload shard not loaded | Verify UnloadShard for unknown model returns failure. | Empty servicer. | model_name="nonexistent". | 1. Call `UnloadShard`. | success=False. |

## Integration Tests

### Heartbeat Flow (`tests/integration/test_heartbeat_flow.py`)

Tests the full gRPC roundtrip - real server on localhost, real channel, real protobuf serialization. Validates that heartbeats sent over the wire reach the registry correctly.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-HF-01 | FR-NM-01 | TestHeartbeatRoundtrip | Heartbeat acknowledged over gRPC | Verify a heartbeat sent through a real gRPC channel returns acknowledged. | In-process gRPC server running. | Default heartbeat request. | 1. Create stub from channel. 2. Call `stub.Heartbeat()`. | acknowledged=True, heartbeat_interval_ms=2000. |
| TC-HF-02 | FR-NM-02 | TestHeartbeatRoundtrip | Multiple heartbeats same node | Verify multiple heartbeats from the same node produce one registry entry. | In-process gRPC server running. | 3 heartbeats with node_id="node-1". | 1. Send 3 heartbeats with same node_id. 2. Call `registry.list_nodes()`. | 1 node in registry. |
| TC-HF-03 | FR-NM-02 | TestHeartbeatRoundtrip | Two different nodes | Verify heartbeats from two nodes create two registry entries. | In-process gRPC server running. | node-1 and node-2 with different IPs. | 1. Send heartbeat for node-1. 2. Send heartbeat for node-2. 3. Call `registry.list_nodes()`. | 2 nodes, IDs {"node-1", "node-2"}. |
| TC-HF-04 | FR-NM-02 | TestHeartbeatRoundtrip | Heartbeat data reaches registry | Verify all fields survive gRPC serialization and reach the registry, including inference_port. | In-process gRPC server running. | Full heartbeat request with all fields set, inference_port=50052. | 1. Send heartbeat. 2. Call `registry.get_node("node-1")`. | All fields match request, status="available", inference_port=50052. |
| TC-HF-05 | FR-NM-04 | TestReportStatusRoundtrip | ReportStatus known node over gRPC | Verify ReportStatus returns correct fields over a real gRPC channel. | In-process gRPC server, one node registered. | node_id="node-1". | 1. Send heartbeat. 2. Call `stub.ReportStatus()`. | Fields match, status=AVAILABLE. |
| TC-HF-06 | FR-NM-04 | TestReportStatusRoundtrip | ReportStatus unknown node raises | Verify ReportStatus raises gRPC NOT_FOUND for unknown nodes. | In-process gRPC server, empty registry. | node_id="nonexistent". | 1. Call `stub.ReportStatus()`. | Raises RpcError with code NOT_FOUND. |

### Failure Detection and End-to-End (`tests/integration/test_failure_detection.py`)

Tests the full orchestrator lifecycle: gRPC server + reaper thread + HeartbeatClients, all in-process. Short intervals (0.1s heartbeat, 0.3s timeout, 0.05s reaper) keep tests fast.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-FD-05 | FR-NM-03 | TestFailureDetection | Node drops after missed heartbeats | Verify a node that stops heartbeating is marked unavailable by the reaper. | Full orchestrator running (server + reaper). | node_id="node-1", interval=0.1s, timeout=0.3s. | 1. Start orchestrator. 2. Start client. 3. Wait 0.3s, verify available. 4. Stop client. 5. Wait for timeout. | Node status is "unavailable". |
| TC-FD-06 | FR-NM-05 | TestFailureDetection | Node rejoins after restart | Verify a previously unavailable node is restored when it resumes heartbeating. | Full orchestrator running, one node previously reaped. | node_id="node-1", interval=0.1s. | 1. Start client, wait, stop. 2. Wait for reap. 3. Start new client with same node_id. 4. Wait 0.3s. | Node status is "available". |
| TC-E2E-01 | FR-NM-01, FR-NM-02, FR-NM-03 | TestEndToEnd | Two workers, one dies | Sprint acceptance test: two workers register, one is killed, the dead one is reaped, the live one stays available. | Full orchestrator running. | worker-1 and worker-2, interval=0.1s. | 1. Start both clients. 2. Wait 0.3s, verify 2 available. 3. Stop worker-2. 4. Wait for timeout. | worker-1 available, worker-2 unavailable. |

### Model Loading (`tests/integration/test_model_loading.py`)

Tests the full HuggingFace model loading and inference pipeline with the real Mamba-130M model. Marked `@pytest.mark.slow`. Module-scoped fixture loads the model once, shared across tests.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-MLI-01 | FR-IE-01 | TestModelLoading | Model handle fields | Verify the loaded handle has correct name, non-None model/tokenizer, and non-negative memory. | Model loaded via fixture. | mamba-130m manifest. | 1. Check handle fields. | name="mamba-130m", model/tokenizer not None, memory_mb >= 0. |
| TC-MLI-02 | FR-IE-02 | TestEndToEndInference | Tokenize returns tensors | Verify tokenize returns input_ids and attention_mask as 2D int64 tensors. | Model loaded via fixture. | prompt="Hey how are you doing?". | 1. Call `tokenize()`. 2. Check types and shapes. | Both 2D, same shape, int64. |
| TC-MLI-03 | FR-IE-01 | TestEndToEndInference | Generate returns string | Verify generation produces a non-empty string. | Model loaded via fixture. | Same prompt, max_new_tokens=10. | 1. Tokenize. 2. Generate. | Non-empty string. |
| TC-MLI-04 | FR-IE-01 | TestEndToEndInference | Reference output match | Verify deterministic output matches the known reference. | Model loaded via fixture. | Same prompt, max_new_tokens=30. | 1. Tokenize. 2. Generate. 3. Compare. | Exact match with reference string. |
| TC-MLI-05 | NFR-04 | TestEndToEndInference | No warnings on stderr | Verify no warning messages appear on stderr during inference. | None (runs in subprocess). | Same prompt. | 1. Run inference in subprocess. 2. Filter stderr for warnings. | No warning lines (progress bars allowed). |

### Inference Flow (`tests/integration/test_inference_flow.py`)

Tests the full inference gRPC roundtrip with real Mamba-130M model. Marked `@pytest.mark.slow`. Module-scoped fixture starts an in-process gRPC server with InferenceServiceServicer.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-IF-01 | FR-IE-01, FR-IE-04 | TestInferenceRoundtrip | Load and run | Load model via gRPC, run inference, verify non-empty output. | In-process gRPC server. | mamba-130m, token IDs, generate_mode=True. | 1. LoadShard. 2. RunShard. | Both succeed, output_tensor non-empty, latency > 0. |
| TC-IF-02 | FR-IE-04 | TestInferenceRoundtrip | Run without load | RunShard before LoadShard raises NOT_FOUND. | In-process gRPC server, no model loaded. | model_name="nonexistent-model". | 1. RunShard. | Raises RpcError with NOT_FOUND. |
| TC-IF-03 | FR-IE-01 | TestInferenceRoundtrip | Unload after load | UnloadShard removes the model from the servicer. | In-process gRPC server, model loaded. | model_name="unload-test-model". | 1. LoadShard. 2. UnloadShard. | success=True, model gone from servicer. |
