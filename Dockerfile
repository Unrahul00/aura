# ── AuraStream Backend ────────────────────────────────────────────────────────
FROM python:3.12-slim

# Install ffmpeg (yt-dlp uses it for audio post-processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/main.py .

# Copy application
COPY backend/main.py .
COPY backend/requirements.txt .

# Non-root user for security
RUN useradd -m -u 1001 aura && chown -R aura:aura /app
USER aura

EXPOSE 8000

# Keep yt-dlp updated at startup (YouTube frequently changes extraction logic)
CMD ["sh", "-c", "pip install -q --upgrade yt-dlp && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2"]
