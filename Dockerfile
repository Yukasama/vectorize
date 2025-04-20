# syntax=docker/dockerfile:1

ARG PYTHON_LATEST_VERSION=3.13
ARG PYTHON_VERSION=${PYTHON_LATEST_VERSION}.3

# ----------------------------------------------
# Stage 1: Builder
# ----------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_LATEST_VERSION}-bookworm-slim AS builder
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


# ----------------------------------------------
# Stage 2: Runtime
# ----------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
WORKDIR /app

# For real time logs
ENV PYTHONUNBUFFERED=1

# Create user with no home dir and no login
RUN groupadd --system appuser && useradd  --system \
            --gid appuser \
            --no-create-home \
            --shell /usr/sbin/nologin \
            appuser

# Copy source code into workdir
COPY --from=builder --chown=appuser:appuser /app ./
RUN find /app -type d -exec chmod 550 {} \; && \
    find /app -type f -exec chmod 440 {} \; && \
    chmod -R 770 /app/data /app/logs 2>/dev/null || true

# Drop privileges
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
STOPSIGNAL SIGINT

CMD ["uvicorn", "txt2vec.app:app", "--host", "0.0.0.0", "--port", "8000"]
