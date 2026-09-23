FROM python:3.11-slim
ARG EXTRAS="infra"
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY epoch ./epoch
COPY configs ./configs
RUN pip install -e ".[${EXTRAS}]"
EXPOSE 8000
CMD ["uvicorn", "epoch.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
