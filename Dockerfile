FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/requirements.txt
COPY data_clean_env/pyproject.toml /app/data_clean_env/pyproject.toml
COPY data_clean_env/server /app/data_clean_env/server
COPY data_clean_env/*.py /app/data_clean_env/
COPY data_clean_env/tasks /app/data_clean_env/tasks

RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -e /app/data_clean_env

# Runtime environment configuration
ENV DATA_CLEAN_TASKS_PATH=/app/data_clean_env/tasks
ENV DATA_CLEAN_MAX_STEPS=30

EXPOSE 8000

CMD ["uvicorn", "data_clean_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
