FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY server /app/server
COPY tasks /app/tasks
COPY models.py /app/models.py
COPY client.py /app/client.py
COPY __init__.py /app/__init__.py

RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -e /app

# Runtime environment configuration
ENV DATA_CLEAN_TASKS_PATH=/app/tasks
ENV DATA_CLEAN_MAX_STEPS=30

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
