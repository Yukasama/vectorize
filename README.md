# Vectorize | AWP Text Embedding Service for Robert Bosch GmbH

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)

  - [Dependency Management](#dependency-management)
    - [Installation](#installation)
    - [Create `.env` File](#create-env-file)
  - [Start Server](#start-server)
  - [Run Tests](#run-tests)
  - [Build Docker Image](#build-docker-image)

- [Workflow](#workflow)

  - [Run CI Locally](#run-ci-locally)
    - [Install `act`](#install-act)
    - [Run `act`](#run-act)

- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Contributing](#contributors)
- [License](#license)

## Introduction

**vectorize** is a self-hosted text embedding service that makes it easy to upload your own corpora, train and evaluate embedding models, and generate synthetic datasets from existing data. Built on FastAPI and PyTorch with SQLModel for persistence, it exposes RESTful endpoints to manage the full lifecycle of text embedding workflows.

The project uses the `uv` tool for seamless dependency management and environment isolation, combined with GitHub Actions for CI and Locust-based load testing to ensure reliability at scale. Packaged in Docker and configured via `.env`, vectorize is designed for both rapid prototyping and production deployment, offering a unified, extensible platform for NLP experimentation and integration.

## Setup

### Dependency management

#### Installation

```sh
# Install uv for command line (MacOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh

# Install uv for command line (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```sh
# Install dependencies and generate lock file
uv sync

## Update dependencies
uv sync --upgrade

# Add dependencies
uv add <package>

# Remove dependencies
uv remove <package>
```

##### Copy .env.example file and rename it to .env

```sh
# .env
DATABASE_URL=sqlite+aiosqlite:///./app.db
SONAR_TOKEN=your-sonar-token
CLEAR_DB_ON_RESTART=1
LOG_LEVEL=DEBUG
TZ=Europe/Berlin
```

Note: Do **not** remove the `.env.example`-File.

#### Fix lock file

Error: Failed to parse `uv.lock`

1. Delete `uv.lock` file
2. Run `uv sync` or `uv lock`

Note: Do **not** edit the `uv.lock`-File yourself.

### Start server

```sh
uv run app
```

### Run linter

```sh
# Run ruff over everything
ruff check .

# Run ruff over specific folder
ruff check src/vectorize/datasets

# Run pyright (Pylance)
pyright
```

#### Run SonarQube

```bash
# 0. Install sonar-scanner
brew install sonar-scanner # (for Mac/Linux)
https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner # (for Windows)

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
```

### Run tests

```sh
# Run all tests
uv run pytest

# Run loadtest with locust
# It's recommended to put LOG_LEVEL=INFO in your .env
uvx locust -f scripts/locust.py

# Run headless locust
# -u for users, -r for rate of spawning users until max
uvx locust -f scripts/locust.py --host=https://localhost/v1 --headless -u 1 -r 1
```

### Build Docker Image

```sh
# Build Docker image
docker build -t vectorize:1.0.0-prod .
```

### Run Grafana

```sh
# --build flag so that newest image is used. If you already have the newest, you can remove the flag.
docker compose up --build
```

## Workflow

### Run CI locally

#### Install act

```sh
# Install uv for command line (MacOS/Linux)
brew install act

# Install uv for command line (Windows)
scoop install act
```

#### Run act

```sh
# Run all CI workflows locally
act

# Or run one specified CI
act -W '.github/workflows/main.yaml'
```

Note: If a CI relies on `GITHUB_TOKEN`, you need to run:

```sh
# You need to have the GitHub CLI installed
act -s GITHUB_TOKEN="$(gh auth token)"
# Plus other arguments
```

Note 2: If a CI uploads or downloads artifacts, you need this flag:

```sh
# This will create the artifacts on your file system
act --artifact-server-path $PWD/.artifacts.
# Plus other arguments
```

```sh
# This is an example showing how to run the Main CI
act -s GITHUB_TOKEN="$(gh auth token)" --artifact-server-path $PWD/.artifacts. -W '.github/workflows/main.yaml'
```

#### Analyze server start with startup.prof

```sh
# Create startup.prof file
python -m cProfile -o startup.prof src/vectorize/app.py

# Start a snakeviz server and look into it
uv run snakeviz startup.prof
```

## Project structure

## Usage examples

## Contributors

Thanks to everyone who’s helped this project!

<!-- markdownlint-disable MD033 MD045 -->

| Contributor                                    | GitHub                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| [@Anselm Böhm](https://github.com/Dosto1ewski) | <img src="https://avatars.githubusercontent.com/Dosto1ewski" width="32" /> |
| [@Botan Coban](https://github.com/BtnCbn)      | <img src="https://avatars.githubusercontent.com/BtnCbn" width="32" />      |
| [@Yukasama](https://github.com/yukasama)       | <img src="https://avatars.githubusercontent.com/yukasama" width="32" />    |
| [@Manuel Dausmann](https://github.com/domoar)  | <img src="https://avatars.githubusercontent.com/domoar" width="32" />      |
| [@Yannic Jahnke](https://github.com/Yannjc)    | <img src="https://avatars.githubusercontent.com/Yannjc" width="32" />      |

<!-- markdownlint-enable MD033 MD045 -->

## License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The Apache License 2.0 is a permissive open‐source license maintained by the Apache Software Foundation. It allows you to freely use, modify, distribute, and sublicense the software, whether in open-source or proprietary projects. Unlike more restrictive copyleft licenses, it does not require derivative works to be distributed under the same license, making it a popular choice for both individual developers and commercial organizations.

Key features of the Apache License 2.0 include:

- **Patent Grant:** Contributors to the project grant you a perpetual, worldwide, royalty‐free license under their patent rights that cover their contributions.
- **Attribution and NOTICE:** You must include a copy of the license and preserve any existing NOTICE file in distributions, ensuring proper attribution to the original authors.
- **No Trademark Rights:** The license does not grant rights to use the project’s trademarks, logos, or branding.

By choosing the Apache License 2.0, we ensure maximum freedom for users and downstream projects, while offering strong legal protections around patents and contributions. Please refer to the full text of the license at the link above for complete terms and conditions.
