FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both backend and frontend so backend can serve frontend/index.html
COPY backend/main.py ./backend/
COPY frontend/index.html ./frontend/

# (Optional) sanity check at container build time; will fail build if files are missing
RUN test -f ./backend/main.py && test -f ./frontend/index.html



EXPOSE 8000

CMD ["sh", "-c", "pip install -q --upgrade yt-dlp && uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
