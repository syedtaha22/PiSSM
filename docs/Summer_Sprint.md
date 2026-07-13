# PiSSM: Summer Sprint Plan

## Objective

The goal of this sprint is to produce a working vertical slice of PiSSM: Mamba inference running across at least two nodes, both interfaces functional at a basic level, and initial benchmark data in hand. The formal FYP semester should begin with working hardware, a proven architecture, and real latency numbers rather than a blank slate.

At the end of this sprint, a user should be able to submit a Mamba-130M checkpoint via the TUI, run inference against it on a live multi-node cluster, and receive output -- without touching any Python directly. Latency benchmarks comparing single-node and two-node execution will exist as a CSV.

---

## Cluster Communication Layer

The first piece of the system to build is the heartbeat and node registry -- not inference, not interfaces. Each Pi runs a daemon that sends a gRPC ping to the orchestrator every 2 seconds. The orchestrator maintains a registry tracking each node's ID, IP address, available RAM, and last-seen timestamp. A node that misses three consecutive pings is marked unavailable.

This requires writing `.proto` files for `NodeService` (heartbeat, status reporting) and a stub `InferenceService` to be populated in the next phase. The target is to run the daemon on two nodes, bring up the orchestrator, and observe both nodes appear in the registry. Killing one daemon should cause it to drop from the registry within 6 seconds.

This is the most critical component to get right. Everything else depends on it.

---

## Mamba Inference on a Single Node

With the cluster skeleton stable, the next step is getting Mamba-130M running correctly on a single node. This involves writing the model loader: code that reads a manifest YAML, locates the checkpoint, instantiates the model in PyTorch, and executes a forward pass. Output is validated against a reference run on a development machine.

The node is then profiled: forward pass latency at 128 tokens, at 512 tokens, and peak memory usage under each. These numbers are the single-node baseline against which all future comparisons are made and must be recorded before proceeding.

Alongside the loader, the inference daemon is written: the worker process that receives a `RunShard` gRPC call carrying serialized input tensors, runs its assigned model layers, and returns serialized output activations.

---

## Layer Sharding Across Two Nodes

Dispatch Rule 2 is implemented in the orchestrator: given a model's layer count and the number of available nodes, layers are split into contiguous groups with one group assigned per node. For Mamba-130M at 24 layers across 2 nodes, node 0 takes layers 0-11 and node 1 takes layers 12-23.

The inference pipeline runs as follows:

1. Orchestrator receives a run request
2. Orchestrator sends `LoadShard` to each worker with the checkpoint path and assigned layer range
3. Workers load their layers into memory
4. Orchestrator sends the input to node 0
5. Node 0 runs its layers and forwards activations to node 1
6. Node 1 runs its layers and returns the result to the orchestrator
7. Orchestrator decodes and returns the result to the user

Two-node latency is measured and compared against the single-node baseline. The difference is the first empirical data point on pipeline parallelism overhead over Gigabit Ethernet.

---

## Interfaces

The TUI is built in Textual with a two-panel layout: a status panel showing the live node list and cluster RAM utilization, and a command panel for user input. The commands `listn`, `compile <manifest>`, `run <model> "<input>"`, and `status` are implemented. The TUI communicates with the orchestrator over FastAPI HTTP.

The WebUI is a React frontend backed by FastAPI. The routes `GET /nodes`, `POST /models`, `GET /models`, and `POST /infer` are defined first. The frontend provides a dashboard with per-node status cards, a model submission form, and an inference panel with text input and output display.

Both interfaces are integrated against the live two-node cluster rather than developed in isolation.

---

## Out of Scope

The following are full system requirements deferred to the FYP semester:

- Automatic re-dispatch on node failure (failure detection is in scope; re-dispatch is not)
- S4 model loader
- LLM support (TinyLlama, Phi-2)
- Quantization
- Topology YAML editor
- Benchmark mode with aggregate statistics and CSV export
- Extended TUI commands (`logs`, `vim topology.yaml`)

---

## An Open Decision

Before the inference daemon is finalized, a decision is needed on shard residency: whether model shards are loaded into worker RAM once and held resident, or reloaded per inference request. Resident shards are faster for repeated inference on the same model; on-demand loading is more flexible when the active model changes frequently. The profiling data from single-node testing will inform this decision and it should be settled before multi-node work begins.

---

## References

- gRPC Python quickstart: https://grpc.io/docs/languages/python/quickstart/
- Protocol Buffers language guide (proto3): https://protobuf.dev/programming-guides/proto3/
- Mamba-130M checkpoint: https://huggingface.co/state-spaces/mamba-130m
- Textual documentation: https://textual.textualize.io/
