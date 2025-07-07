FROM python:3.11-slim-buster as builder

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files (adjust if Streamlit app has its own pyproject.toml/uv.lock)
# For simplicity, assuming it might share some or needs its own for Streamlit
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies, including Streamlit
# You might need to add 'streamlit' to your pyproject.toml or install it directly
RUN uv pip install --system -e . streamlit

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
# This will copy app.py and any other necessary files for your Streamlit app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit application
# 'app.py' assumes your Streamlit application file is named 'app.py' in the root
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
