FROM python:3.12-slim

WORKDIR /app

# 0. Install system dependencies (libgomp1 for LightGBM, curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# 1. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy FastAPI app and pipeline code
COPY app.py .
COPY titanic_pipeline.py .

# 3. Copy models INTO the image (THIS WAS MISSING)
COPY models /app/models

EXPOSE 8000

# Container-level healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 4. Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
