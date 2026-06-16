# ============================================================
# Dockerfile — Jio Attrition Intelligence System
# Single image used by both the API and Dashboard services
# (docker-compose overrides the CMD for the dashboard container)
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System deps for some ML libs (xgboost, lifelines) that need build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first — cached layer, only rebuilds if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

# Create dirs that the app writes to at runtime (in case they're not in the repo)
RUN mkdir -p logs data/synthetic

EXPOSE 8000
EXPOSE 8501

# Default command — overridden by docker-compose for the dashboard service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
