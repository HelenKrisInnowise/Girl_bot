# FROM python:3.11-slim-buster

# RUN pip install --no-cache-dir uv

# WORKDIR /app

# COPY pyproject.toml .
# COPY uv.lock .

# RUN uv pip install --system -e .

# COPY . .

# EXPOSE 8501

# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Base image with Python 3.11
FROM python:3.11-slim-buster as builder

# Install uv (ultra-fast pip replacement)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# First copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies with uv (system-wide)
RUN uv pip install --system -e .

# --- Production stage ---
FROM python:3.11-slim-buster

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command (with auto-reload for development)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]