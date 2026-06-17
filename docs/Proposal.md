# PiSSM: Distributed State Space Model Inference on Commodity Edge Hardware

## Overview

Large AI models typically require expensive server-grade hardware to run. GPUs with tens of gigabytes of memory, cloud compute costing hundreds of dollars per month, and specialized infrastructure are taken for granted in modern AI deployments. This project asks a different question: what if you had neither?

PiSSM is a distributed inference system designed to run modern AI models across a cluster of Raspberry Pi 5 units. Six single-board computers, each with 4 GB of RAM and connected over Gigabit Ethernet, form a 24 GB cluster capable of serving inference requests for State Space Models (SSMs) and small language models. The system handles everything automatically: detecting which nodes are available, deciding how to split a model across them, routing data through the pipeline, and recovering if a node fails. The user provides a model checkpoint and a short configuration file. The system does the rest.

The project is not a proof of concept. It is a complete, usable system with a professional terminal interface for power users and a browser-based dashboard for general use.

## Motivation

State Space Models, particularly Mamba and S4, represent a significant development in sequence modeling. Unlike transformers, SSMs scale linearly with sequence length rather than quadratically, making them attractive for long-context tasks and deployment on memory-constrained hardware. Their recurrent inference structure means that, once a model is loaded, each new token or time step requires only a small fixed computation -- the hidden state carries forward, not the entire context window.

This makes SSMs a natural fit for edge inference. The question is whether a real system can be built that takes advantage of this efficiently, and whether commodity hardware like the Raspberry Pi 5 can be turned into a coherent inference platform rather than just an embedded curiosity.

The broader relevance is real. Not every deployment context has access to cloud compute. Medical devices, agricultural monitoring systems, offline educational tools, and remote industrial sensors all represent environments where AI inference must happen locally, on constrained hardware, reliably. A system that demonstrates distributed SSM inference on six Raspberry Pis is a direct proof of that capability.

## Target Architectures

**S4 (Structured State Spaces for Sequences)** is the foundational architecture. S4 models a sequence as a continuous-time dynamical system discretized for practical computation. The core operation is a convolution over a learned state matrix, enabling parallelism during training and efficient recurrence at inference time.

**Mamba** refines S4 by making the state space parameters input-dependent, a technique called selective state spaces. This allows the model to filter irrelevant information from the sequence dynamically. Mamba has shown competitive performance with transformers on language tasks at a fraction of the inference cost per token.

**Small transformer LLMs** (TinyLlama at 1.1B parameters, Phi-2 at 2.7B) are included as a secondary target. They serve as a comparison baseline and demonstrate that the system's architecture is not SSM-specific -- any layer-decomposable model can be served.

## System Capabilities

A user interacts with PiSSM through one of two interfaces.

The **terminal interface** (TUI) is built with Textual, a Python framework for professional terminal applications. It supports a command vocabulary: `listn` to view cluster node status, `compile <manifest>` to register a model, `run <model> "<input>"` to execute inference, and `vim topology.yaml` to inspect or override how the model is distributed across nodes. The TUI is designed for power users who want precise control and live visibility into the cluster.

The **web interface** (WebUI) is a React application served by the orchestrator node. It provides a dashboard showing cluster health, a model submission form, an inference panel, and a topology visualization showing which model layers are assigned to which nodes. It is accessible from any device on the local network.

Behind both interfaces, the system runs the following way. Each Pi runs a background daemon that broadcasts its presence and hardware status. The orchestrator node -- one Pi designated as the coordinator -- maintains a live registry of all workers. When a user submits a model and requests inference, the orchestrator consults available memory across nodes, applies a dispatch rule to decide how many nodes to use and how to split the model's layers, assigns shards to workers, and routes the inference request through the pipeline. Activations pass from node to node over gRPC until the final node returns the result to the orchestrator, which returns it to the user.

If a node fails during inference, the orchestrator detects the missing heartbeat, marks the node unavailable, and re-dispatches the request to the remaining cluster. This happens without user intervention.

## Research Contribution

The system produces a structured benchmark dataset as a natural output of operation. Every inference request records end-to-end latency, per-node execution time, activation transfer overhead, and peak memory usage. Across varying model sizes, node counts, and sequence lengths, this data characterizes the performance envelope of distributed SSM inference on edge hardware in a way that does not currently exist in the literature (to the best of our knowledge).

Specific questions the project aims to answer with measured data:

- What is the latency cost of pipeline parallelism across Gigabit Ethernet relative to single-node inference for Mamba and S4 models?
- At what model size does distributing inference across additional nodes yield net latency improvement rather than net degradation?
- How does SSM inference scale with sequence length on this hardware compared to transformer baselines?
- What is the fault recovery overhead when a node fails mid-inference?

These results are targeted for submission to a systems or efficient ML venue: MLSys, or a workshop at NeurIPS or ICLR focused on efficient inference or edge AI.

## Hardware Summary

| Resource | Per Node | Cluster Total |
|----------|----------|---------------|
| RAM | 4 GB | 24 GB |
| Storage | 20 GB SD | 120 GB |
| Network | Gigabit Ethernet | Switched LAN |
| Nodes | -- | 6x RPi 5B |

The cluster can hold models up to approximately 10 billion parameters in fp16 when fully distributed, or up to approximately 20 billion parameters with int8 quantization -- a capability range that covers all current Mamba and S4 variants and a meaningful portion of the open-weight LLM ecosystem.