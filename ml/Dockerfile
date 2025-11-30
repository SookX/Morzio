# ============================================
# Morzio ML Service Dockerfile
# Multi-stage build with CPU-only PyTorch
# Target size: ~500MB (vs 2GB+ with CUDA)
# ============================================

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY api/requirements.txt .

# Create virtual environment and install CPU-only PyTorch
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ============================================
# Stage 2: Runtime (minimal)
# ============================================
FROM python:3.12-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy only necessary application code
COPY api/ ./api/

# VAE model (code + weights only)
COPY vae_annomaly_detection/model/ ./vae_annomaly_detection/model/
COPY vae_annomaly_detection/checkpoints/ ./vae_annomaly_detection/checkpoints/
COPY vae_annomaly_detection/config.yaml ./vae_annomaly_detection/
COPY vae_annomaly_detection/pipeline.py ./vae_annomaly_detection/
COPY vae_annomaly_detection/utils.py ./vae_annomaly_detection/

# Dataset module only
RUN mkdir -p ./vae_annomaly_detection/dataset
COPY vae_annomaly_detection/dataset/dataset.py ./vae_annomaly_detection/dataset/

# Installment policy
COPY dema_revenue/ ./dema_revenue/

# Training features for scaler (8.6MB)
COPY data/training_features.csv ./data/

# Create __init__.py files
RUN touch ./vae_annomaly_detection/__init__.py \
    && touch ./vae_annomaly_detection/dataset/__init__.py \
    && touch ./dema_revenue/__init__.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
