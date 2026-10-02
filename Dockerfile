FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-postgres.txt

# Copy application code and install the package without moving the pins above.
COPY . .
RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 1000 asylum \
    && mkdir -p /app/data \
    && chown -R asylum:asylum /app
USER asylum

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 0.0.0.0 is required inside the container so the published port is reachable.
# docker-compose.yml refuses to start unless REQUIRE_AUTH and API_KEYS are set.
CMD ["sh", "-c", "python -m alembic upgrade head && python -m uvicorn vivasecuris.aiasylum.api.main:app --host 0.0.0.0 --port 8000"]
