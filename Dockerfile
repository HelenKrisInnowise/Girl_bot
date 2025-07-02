FROM python:3.11-slim-buster

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN uv pip install --system -e .

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]