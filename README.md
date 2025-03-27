# AWP Projekt für Text Embedding Service

## Setup

### Dependency management

```bash
# Install dependencies
poetry install
```

```bash
# Add dependencies
poetry add \<package\>
```

### Start server

```bash
poetry run uvicorn src.txt2vec_service.main:app --reload --reload-dir src
```

### Code analysis

```bash
poetry run black .
```
