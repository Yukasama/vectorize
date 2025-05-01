# AWP Projekt für Text Embedding Service

Table of contents

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

##### Spawn a Shell

You can open an interactive shell with your project’s virtual environment activated — this ensures your venv’s Python, pip, and CLI tools are on your PATH and enabled:

```bash
# Linux
uv run -- bash   # launch Bash
uv run -- sh     # launch POSIX sh

# macOS (default shell is zsh)
uv run -- zsh

# Windows PowerShell
uv run -- pwsh -NoExit
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

## Project structure

## Usage examples

## Contributors

Thanks to everyone who’s helped this project!

<!-- markdownlint-disable MD033 MD045 -->
| Contributor                                        | GitHub                                                                       |
|----------------------------------------------------|------------------------------------------------------------------------------|
| [@Anselm Böhm](https://github.com/Dosto1ewski)     | <img src="https://avatars.githubusercontent.com/Dosto1ewski" width="32" />   |
| [@Botan Coban](https://github.com/BtnCbn)          | <img src="https://avatars.githubusercontent.com/BtnCbn" width="32" />        |
| [@Christopher Claus](https://github.com/yukasama)  | <img src="https://avatars.githubusercontent.com/yukasama" width="32" />      |
| [@Manuel Dausmann](https://github.com/domoar)      | <img src="https://avatars.githubusercontent.com/domoar" width="32" />        |
| [@Yanick Janka](https://github.com/Yannjc)         | <img src="https://avatars.githubusercontent.com/Yannjc" width="32" />        |
<!-- markdownlint-enable MD033 MD045 -->

## License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).  

The Apache License 2.0 is a permissive open‐source license maintained by the Apache Software Foundation. It allows you to freely use, modify, distribute, and sublicense the software, whether in open-source or proprietary projects. Unlike more restrictive copyleft licenses, it does not require derivative works to be distributed under the same license, making it a popular choice for both individual developers and commercial organizations.

Key features of the Apache License 2.0 include:

- **Patent Grant:** Contributors to the project grant you a perpetual, worldwide, royalty‐free license under their patent rights that cover their contributions.  
- **Attribution and NOTICE:** You must include a copy of the license and preserve any existing NOTICE file in distributions, ensuring proper attribution to the original authors.  
- **No Trademark Rights:** The license does not grant rights to use the project’s trademarks, logos, or branding.

By choosing the Apache License 2.0, we ensure maximum freedom for users and downstream projects, while offering strong legal protections around patents and contributions. Please refer to the full text of the license at the link above for complete terms and conditions.  
