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
uv sync

## Update dependencies
uv sync --upgrade

# Add dependencies
uv add <package>

# Remove dependencies
uv remove <package>
```

##### Create .env file

```bash
# .env
DATABASE_URL=sqlite+aiosqlite:///./app.db
```

#### Fix lock file

Error: Failed to parse `uv.lock`

1. Delete `uv.lock` file
2. Run `uv sync` or `uv lock`

Note: Do **not** edit the `uv.lock`-File yourself.

### Start server

```bash
uv run app
```

### Run linter

```bash
# Run ruff over everything
uv run ruff check .

# Run ruff over specific folder
uv run ruff check src/txt2vec/datasets
```

#### Run SonarQube

````bash
# 0. Install sonar-scanner (for Mac/Linux)
brew install sonar-scanner

# 0. Install sonar-scanner (for Windows)
Go to https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner and install

# Only once: Create .env under resources/sonarqube

```bash
# .env
TZ=Europe/Berlin
```

# 1. Run docker container

cd resources/sonarqube
docker compose up

# Only once: Get token from SonarQube

# 1. Go to http://localhost:9000

# 2. Account > Settings

# 3. Generate Global Token with 'No expiration'

# 4. Copy into .env under SONAR_TOKEN

# 2. Run sonar scan

uv run scripts/sonar_scan.py

````

### Run tests

```bash
# Run all tests
uv run pytest

# Run loadtest with locust
uvx locust -f scripts/locust.py
```

### Build Docker Image

```bash
# Build Docker image
docker build -t txt2vec:1.0.0-prod .
```

## Workflow

### Run CI locally

#### Install act

```bash
# Install uv for command line (MacOS/Linux)
brew install act

# Install uv for command line (Windows)
scoop install act
```

#### Run act

```bash
# Run all CI workflows locally
act

# Or run one specified CI
act -W '.github/workflows/main.yaml'
```

Note: If a CI relies on `GITHUB_TOKEN`, you need to run:

```bash
# You need to have the GitHub CLI installed
act -s GITHUB_TOKEN="$(gh auth token)"
# Plus other arguments
```

Note 2: If a CI uploads or downloads artifacts, you need this flag:

```bash
# This will create the artifacts on your file system
act --artifact-server-path $PWD/.artifacts.
# Plus other arguments
```

```bash
# This is an example showing how to run the Main CI
act -s GITHUB_TOKEN="$(gh auth token)" --artifact-server-path $PWD/.artifacts. -W '.github/workflows/main.yaml'
```
