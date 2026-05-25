FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both backend and frontend so backend can serve frontend/index.html
COPY backend/main.py ./backend/
COPY frontend/index.html ./frontend/


EXPOSE 8000

CMD ["sh", "-c", "pip install -q --upgrade yt-dlp && uvicorn main:app --host 0.0.0.0 --port 8000"]