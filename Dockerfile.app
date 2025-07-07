FROM python:3.11-slim-buster as builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN uv pip install --system -e . streamlit

FROM python:3.11-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
