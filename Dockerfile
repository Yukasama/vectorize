# ----------------------------------------------
# Stage 1: Installation
# ----------------------------------------------
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (for local builds, not needed in final image)
RUN pip install --no-cache-dir uv

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock* README.md /app/

# Install dependencies to a temporary directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

    
# ----------------------------------------------
# Stage 2: Run
# ----------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user and group
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app

# Set permissions
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Use uvicorn to run the FastAPI app
CMD ["uvicorn", "src.txt2vec.app:app", "--host", "0.0.0.0", "--port", "8000"]