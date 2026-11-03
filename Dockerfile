# === MathVision AI — production image ===
# Serves the FastAPI backend by default. The Streamlit frontend can be
# deployed as a second instance of this same image with a different
# CMD (see docker-compose usage in README), or separately on Streamlit
# Cloud pointed at this repo's frontend/app.py.

FROM python:3.11-slim AS base

# System dependencies:
#   - libgl1/libglib2.0-0: required by opencv-python-headless at import time
#   - ffmpeg: required by openai-whisper (audio decoding for voice doubts)
#   - fonts-dejavu-core: used by scripts/generate_problems.py's PIL rendering
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first to keep the image small (no CUDA needed
# for inference-scale TrOCR/Whisper on modest traffic); then the rest.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Optional: bake the fine-tuned TrOCR weights (2.2 GB) into the image.
# The OCR model needs >= 4 GB RAM at runtime, so keep this off for small hosts.
#   DOCKER_BUILDKIT=1 docker build --build-arg DOWNLOAD_MODEL=1 \
#       --secret id=hf_token,src=hf_token.txt -t mathvision-ai:ocr .
ARG DOWNLOAD_MODEL=0
RUN --mount=type=secret,id=hf_token \
    if [ "$DOWNLOAD_MODEL" = "1" ]; then \
        python scripts/download_model.py --token "$(cat /run/secrets/hf_token)"; \
    fi

RUN mkdir -p /app/logs /app/data/student_profiles /app/chromadb_store

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Render/Heroku-style $PORT support, defaulting to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
