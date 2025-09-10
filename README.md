# Natural Language Processing

This repo contains materials for the course [ETE387 Natural Language Processing](https://liu-nlp.ai/ete387/).

## Setup

We use [`uv`](https://github.com/astral-sh/uv) to manage Python environments and dependencies.

* Python ≥ 3.13
* `uv` installed ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

Create and sync a virtual environment with the exact locked dependencies:

```bash
uv sync
```

This will:

* Create a `.venv/` directory (if not already present)
* Install all dependencies exactly as specified in `uv.lock`

Once synced, activate the virtual environment to run commands in the context of the project:

```bash
source .venv/bin/activate   # Linux and macOS
.venv\Scripts\activate      # Windows (PowerShell: .venv\Scripts\Activate.ps1)
```

You will also be able to select the environment in VS Code.
