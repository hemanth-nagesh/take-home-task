# =============================================================================
# Stage 1 — builder: install dependencies into /install
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install pip dependencies into a prefix directory so the runtime stage
# can copy only the installed packages (no build tools, no cache).
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================================
# Stage 2 — runtime: lean image with non-root user
# =============================================================================
FROM python:3.12-slim AS runtime

# Install wget for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user
RUN addgroup --gid 1000 appuser \
 && adduser  --uid 1000 --gid 1000 --no-create-home --disabled-password appuser

WORKDIR /app

# Copy application source (all packages live at the repo root)
COPY . .
RUN chown -R appuser:appuser /app

# Ensure common/ and all top-level packages are importable
ENV PYTHONPATH=/app

# Drop privileges
USER appuser

# Expose all three service ports
EXPOSE 8000 8001 8002

# Default: run the orchestrator (overridden per-service in docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
