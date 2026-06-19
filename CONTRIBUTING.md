# Contributing Guidelines

### Purpose

This document defines the conventions for forking, branching, commits, pull requests, and code style across this repository. Consistency and clarity are expected from all contributors.

---

## 1. Workflow

This repository does not accept direct pushes. All contributions must come through a fork.

1. Fork the repository to your GitHub account
2. Clone your fork locally
3. Create a branch for your work
4. Push to your fork and open a pull request targeting `main` on this repository

---

## 2. Branching

Branches should reflect the nature and scope of the work. Use the following scheme:

| Branch | Purpose |
|--------|---------|
| `feat/<description>` | New functionality |
| `fix/<description>` | Bug fixes |
| `exp/<topic>` | Experimental or research work |
| `docs/<topic>` | Documentation only |

Keep each branch focused on a single topic. One branch, one purpose.

---

## 3. Commit Messages

Each commit should represent one logical, self-contained change. Use the following format:

```
<type>: <short imperative description>

[optional body]
```

### Allowed Types

| Type | Description |
|------|-------------|
| `feat` | New functionality |
| `fix` | Bug correction |
| `perf` | Performance improvement |
| `refactor` | Structural change with no behavior change |
| `docs` | Documentation only |
| `chore` | Dependencies, configuration, tooling |
| `research` | Benchmarks, experiments, hypothesis testing |
| `exp` | Prototype or exploratory work |

### Rules

- Subject line must be under 72 characters
- Use imperative mood: "Add", "Fix", "Update" - not "Added" or "Fixes"
- No trailing period on the subject line
- Use the body to explain *why*, not *what*, when the subject alone is insufficient
- Avoid vague messages like "update files" or "changes done"

### Examples

```
feat: add mamba-370m loader with manifest validation
fix: resolve activation tensor shape mismatch on shard boundary
perf: cache loaded shard in worker memory after first inference
refactor: split dispatch engine into planner and executor modules
research: benchmark pipeline parallelism overhead across 2 and 4 nodes
docs: update manifest format specification in SRS
```

---

## 4. Pull Requests

- Fill out the pull request template in full
- Keep PRs focused - one feature or fix per PR
- Changes touching the worker daemon, dispatch engine, or gRPC layer must be tested on Raspberry Pi hardware before submission
- All PRs require at least one review before merge

---

## 5. Code Style

### 5.1 Python

The project follows PEP 8. All Python code must be formatted with `black` before committing:

```bash
black .
```

**Naming conventions:**

| Construct | Convention |
|-----------|------------|
| Functions and variables | `snake_case` |
| Classes | `PascalCase` |
| Constants | `UPPER_CASE` |
| Private members | `_prefixed` |

All public functions and classes must have docstrings. For non-trivial functions, include parameter descriptions and return values.

```python
def load_shard(manifest_path: str, layer_range: tuple[int, int]) -> nn.Module:
    """
    Load a contiguous range of model layers from a checkpoint.

    Parameters
    ----------
    manifest_path : str
        Path to the model manifest YAML file.
    layer_range : tuple[int, int]
        Inclusive start and end layer indices to load.

    Returns
    -------
    nn.Module
        The loaded model shard ready for inference.
    """
```

### 5.2 Protocol Buffers

- One service definition per `.proto` file
- Field names in `snake_case`
- Keep service interfaces minimal - only expose what the orchestrator and workers require at the gRPC boundary

---

## 6. Best Practices

| Area | Do | Avoid |
|------|----|-------|
| Branching | Use descriptive names like `exp/dispatch-profiling` | Pushing directly to `main` |
| Commits | Keep each commit atomic and focused | Bundling unrelated changes |
| Messages | Follow the format and use imperative mood | Vague or incomplete subjects |
| Pull Requests | Test thoroughly before opening | Submitting untested hardware changes |
| Code | Format, document, and lint before committing | Leaving debug code or commented-out blocks |

---

## 7. Reporting Issues

Open a GitHub issue with a clear description of the problem, reproduction steps, and the relevant environment: OS, Python version, and whether the issue occurs on a Pi or a development machine.
