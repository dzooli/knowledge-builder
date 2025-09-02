FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy project manifests from repo root (build context will be set to repo root)
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

RUN pip install uv && \
    uv venv && \
    uv sync --frozen

# Ensure venv binaries (e.g., mcp-neo4j-memory) are on PATH for subprocess calls
ENV PATH="/app/.venv/bin:${PATH}"

# Copy entire importer source directory structure
COPY importer/src/ /app/src/

# Add src directory to Python path for proper imports
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/app/.venv/bin/python","/app/src/main.py"]
