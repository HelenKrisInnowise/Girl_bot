FROM python:3.11-slim-buster

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN uv pip install -e .

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]