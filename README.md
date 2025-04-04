# AWP Projekt für Text Embedding Service

## Setup

### Dependency management

#### Installation

```bash
# Install uv for command line (MacOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh

# Install uv for command line (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Install dependencies and generate lock file
uv sync --all-groups
```

#### Modification

```bash
# Add dependencies
uv add <package>

# Update dependencies
uv update

# Remove dependencies
uv remove <package>
```

#### Fix lock file

Error: Failed to parse `uv.lock`

1. Delete `uv.lock` file
2. Run `uv sync --all-groups` or `uv lock`

Note: Do not edit the `uv.lock`-File yourself.

### Start server

```bash
uv run txt2vec_service
```

### Build

```bash
# Build project
uv build
```
