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
| TC-NR-01 | FR-NM-02 | TestNodeRegistration | Register new node | Verify that registering a node stores all fields correctly and sets the initial status to available. | Empty registry with mock clock at t=100.0. | node_id="node-1", ip="192.168.1.10", available_ram=3800, total_ram=4096, cpu_count=4, arch="aarch64", os_name="Linux", os_version="6.6.31+rpt-rpi-2712". | 1. Create registry with mock clock at t=100.0. 2. Call `update_node` with test data. 3. Call `get_node("node-1")`. | All fields match input, status="available", last_heartbeat=100.0, first_seen=100.0. |
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
| TC-HS-02 | FR-NM-02 | TestHeartbeatRPC | Heartbeat registers node | Verify that after a heartbeat the node exists in the registry with correct fields. | Empty registry. | node_id="node-1", ip="192.168.1.10", available_ram=3800, total_ram=4096, cpu_count=4, arch="aarch64". | 1. Call `Heartbeat`. 2. Call `registry.get_node("node-1")`. | Node exists with all fields matching, status="available". |
| TC-HS-03 | FR-NM-01 | TestHeartbeatRPC | Heartbeat returns configured interval | Verify that the response carries the heartbeat interval configured on the servicer. | Empty registry, servicer with heartbeat_interval_ms=5000. | Default heartbeat request. | 1. Create servicer with interval=5000. 2. Call `Heartbeat`. | response.heartbeat_interval_ms == 5000. |
| TC-HS-04 | FR-NM-01 | TestHeartbeatRPC | Heartbeat default interval | Verify that the default heartbeat interval is 2000ms. | Empty registry. | Default heartbeat request. | 1. Create servicer with default config. 2. Call `Heartbeat`. | response.heartbeat_interval_ms == 2000. |
| TC-HS-05 | FR-NM-04 | TestReportStatusRPC | ReportStatus returns correct fields | Verify that ReportStatus returns all node fields for a known node. | Registry with one node registered via Heartbeat. | node_id="node-1". | 1. Send Heartbeat. 2. Call `ReportStatus(node_id="node-1")`. | All fields match, status=AVAILABLE. |
| TC-HS-06 | FR-NM-04 | TestReportStatusRPC | ReportStatus unknown node | Verify that ReportStatus sets NOT_FOUND for an unknown node. | Empty registry. | node_id="nonexistent". | 1. Call `ReportStatus(node_id="nonexistent")`. | context.set_code called with NOT_FOUND, context.set_details called. |

### HeartbeatClient (`tests/unit/test_heartbeat_client.py`)

Component under test: `worker.heartbeat.HeartbeatClient` - periodically sends gRPC heartbeats to the orchestrator from a background thread. Each test creates its own in-process gRPC server with a matching heartbeat interval so the server does not override the client's test interval.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-HC-01 | FR-NM-01 | TestHeartbeatClientLifecycle | Client sends heartbeat | Verify the client sends at least one heartbeat after starting. | In-process gRPC server with interval=200ms. | node_id="test-node", interval=0.2s. | 1. Start client. 2. Wait 0.5s. 3. Stop client. 4. Check registry. | Node exists with status="available". |
| TC-HC-02 | FR-NM-01 | TestHeartbeatClientLifecycle | Client stops cleanly | Verify that after stop(), is_running is False. | In-process gRPC server with interval=200ms. | node_id="test-node", interval=0.2s. | 1. Start client. 2. Assert is_running=True. 3. Stop client. | is_running is False. |
| TC-HC-03 | FR-NM-01 | TestHeartbeatClientLifecycle | Client sends multiple heartbeats | Verify heartbeats continue periodically, not just on startup. | In-process gRPC server with interval=200ms. | node_id="test-node", interval=0.2s. | 1. Start client. 2. Wait 1.0s. 3. Read last_heartbeat. 4. Compare to current time. | last_heartbeat is within 0.5s of current time. |
| TC-HC-04 | NFR-02 | TestHeartbeatClientResilience | Client handles server unavailable | Verify the client does not crash when the orchestrator is unreachable. | No server running on target port. | orchestrator_address="localhost:1", interval=0.2s. | 1. Start client. 2. Wait 0.5s. 3. Stop client. | is_running is False, no exceptions. |

## Integration Tests

### Heartbeat Flow (`tests/integration/test_heartbeat_flow.py`)

Tests the full gRPC roundtrip - real server on localhost, real channel, real protobuf serialization. Validates that heartbeats sent over the wire reach the registry correctly.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-HF-01 | FR-NM-01 | TestHeartbeatRoundtrip | Heartbeat acknowledged over gRPC | Verify a heartbeat sent through a real gRPC channel returns acknowledged. | In-process gRPC server running. | Default heartbeat request. | 1. Create stub from channel. 2. Call `stub.Heartbeat()`. | acknowledged=True, heartbeat_interval_ms=2000. |
| TC-HF-02 | FR-NM-02 | TestHeartbeatRoundtrip | Multiple heartbeats same node | Verify multiple heartbeats from the same node produce one registry entry. | In-process gRPC server running. | 3 heartbeats with node_id="node-1". | 1. Send 3 heartbeats with same node_id. 2. Call `registry.list_nodes()`. | 1 node in registry. |
| TC-HF-03 | FR-NM-02 | TestHeartbeatRoundtrip | Two different nodes | Verify heartbeats from two nodes create two registry entries. | In-process gRPC server running. | node-1 and node-2 with different IPs. | 1. Send heartbeat for node-1. 2. Send heartbeat for node-2. 3. Call `registry.list_nodes()`. | 2 nodes, IDs {"node-1", "node-2"}. |
| TC-HF-04 | FR-NM-02 | TestHeartbeatRoundtrip | Heartbeat data reaches registry | Verify all fields survive gRPC serialization and reach the registry. | In-process gRPC server running. | Full heartbeat request with all fields set. | 1. Send heartbeat. 2. Call `registry.get_node("node-1")`. | All fields match request, status="available". |
| TC-HF-05 | FR-NM-04 | TestReportStatusRoundtrip | ReportStatus known node over gRPC | Verify ReportStatus returns correct fields over a real gRPC channel. | In-process gRPC server, one node registered. | node_id="node-1". | 1. Send heartbeat. 2. Call `stub.ReportStatus()`. | Fields match, status=AVAILABLE. |
| TC-HF-06 | FR-NM-04 | TestReportStatusRoundtrip | ReportStatus unknown node raises | Verify ReportStatus raises gRPC NOT_FOUND for unknown nodes. | In-process gRPC server, empty registry. | node_id="nonexistent". | 1. Call `stub.ReportStatus()`. | Raises RpcError with code NOT_FOUND. |

### Failure Detection and End-to-End (`tests/integration/test_failure_detection.py`)

Tests the full orchestrator lifecycle: gRPC server + reaper thread + HeartbeatClients, all in-process. Short intervals (0.1s heartbeat, 0.3s timeout, 0.05s reaper) keep tests fast.

| Test Case ID | Requirement | Test Suite | Title | Description | Pre-conditions | Test Data | Test Steps | Expected Result |
|---|---|---|---|---|---|---|---|---|
| TC-FD-05 | FR-NM-03 | TestFailureDetection | Node drops after missed heartbeats | Verify a node that stops heartbeating is marked unavailable by the reaper. | Full orchestrator running (server + reaper). | node_id="node-1", interval=0.1s, timeout=0.3s. | 1. Start orchestrator. 2. Start client. 3. Wait 0.3s, verify available. 4. Stop client. 5. Wait for timeout. | Node status is "unavailable". |
| TC-FD-06 | FR-NM-05 | TestFailureDetection | Node rejoins after restart | Verify a previously unavailable node is restored when it resumes heartbeating. | Full orchestrator running, one node previously reaped. | node_id="node-1", interval=0.1s. | 1. Start client, wait, stop. 2. Wait for reap. 3. Start new client with same node_id. 4. Wait 0.3s. | Node status is "available". |
| TC-E2E-01 | FR-NM-01, FR-NM-02, FR-NM-03 | TestEndToEnd | Two workers, one dies | Sprint acceptance test: two workers register, one is killed, the dead one is reaped, the live one stays available. | Full orchestrator running. | worker-1 and worker-2, interval=0.1s. | 1. Start both clients. 2. Wait 0.3s, verify 2 available. 3. Stop worker-2. 4. Wait for timeout. | worker-1 available, worker-2 unavailable. |
