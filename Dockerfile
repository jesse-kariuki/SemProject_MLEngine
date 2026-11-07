# FloodGuard API. Build from the repo root: docker build -t floodguard-api .
# .dockerignore limits the build to backend/, so the models and raw data never enter the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/assets ./assets

RUN useradd --create-home floodguard && mkdir /srv/data && chown floodguard /srv/data
USER floodguard

# SQLite is only durable if /srv/data is a persistent volume. Otherwise set DATABASE_URL to a managed Postgres.
ENV DATABASE_URL=sqlite:////srv/data/floodguard.db
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os, urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health/live', timeout=4)"

# Shell form so a platform-assigned $PORT applies. One worker on purpose: the scheduler runs in-process.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
