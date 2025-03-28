# AWP Projekt für Text Embedding Service

## Setup

### Dependency management

#### Installation

```bash
# Install dependencies
poetry install

# Generate or fix lock file
poetry lock

# Check for dependency conflicts
poetry check
```

#### Modification

```bash
# Add dependencies
poetry add <package>

# Update dependencies
poetry update

# Remove dependencies
poetry remove <package>
```

#### Build

```bash
# Build project
poetry build
```

### Start server

```bash
poetry run uvicorn src.txt2vec_service.main:app --reload --reload-dir src
```
