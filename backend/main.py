"""
AuraStream — FastAPI Reverse-Proxy Audio Backend
Phase 1: Stream Engine
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import httpx
import yt_dlp
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aurastream")


# ── LRU In-Memory Cache ───────────────────────────────────────────────────────
class LRUCache:
    """Thread-safe LRU cache with TTL expiration."""

    def __init__(self, capacity: int = 256, ttl: int = 3600):
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._capacity = capacity
        self._ttl = ttl

    def get(self, key: str) -> Optional[dict]:
        if key not in self._cache:
            return None
        entry, ts = self._cache[key]
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry

    def set(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def evict(self, key: str) -> None:
        self._cache.pop(key, None)


# Global caches
_info_cache = LRUCache(capacity=512, ttl=3600)   # track metadata
_url_cache  = LRUCache(capacity=128, ttl=1800)   # resolved stream URLs (shorter TTL)


# ── yt-dlp Configuration Factories ───────────────────────────────────────────
_COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
    "Connection": "keep-alive",
}


def _search_opts() -> dict:
    return {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "youtube_include_dash_manifest": False,
        "http_headers": _COMMON_HEADERS,
        "socket_timeout": 15,
    }


def _stream_opts() -> dict:
    return {
        "format": "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "youtube_include_dash_manifest": False,
        "noplaylist": True,
        "http_headers": _COMMON_HEADERS,
        "socket_timeout": 20,
    }


# ── yt-dlp Async Wrappers ─────────────────────────────────────────────────────

async def _run_ydl(opts: dict, url_or_query: str) -> dict:
    """
    Run yt-dlp extraction in a worker thread to avoid blocking the async loop.
    Returns the raw info dict from yt-dlp.
    """
    def _extract() -> dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url_or_query, download=False)

    return await asyncio.to_thread(_extract)


async def search_tracks(query: str, max_results: int = 20) -> list[dict]:
    """
    Search YouTube Music / YouTube for audio tracks.
    Returns a sanitised list of track metadata dicts.
    """
    cached = _info_cache.get(f"search:{query}:{max_results}")
    if cached:
        log.info("Cache HIT — search: %s", query)
        return cached["results"]

    yt_query = f"ytsearch{max_results}:{query}"
    opts = _search_opts()

    raw = await _run_ydl(opts, yt_query)

    entries = raw.get("entries") or []
    results: list[dict] = []

    for entry in entries:
        if not entry:
            continue
        track = _normalise_entry(entry)
        if track:
            results.append(track)

    _info_cache.set(f"search:{query}:{max_results}", {"results": results})
    return results


async def resolve_stream_url(video_id: str) -> dict:
    """
    Resolves the direct audio stream URL for a given YouTube video ID.
    Returns a dict with `url`, `ext`, `content_type`, and track metadata.
    """
    cached = _url_cache.get(video_id)
    if cached:
        log.info("Cache HIT — stream URL: %s", video_id)
        return cached

    yt_url = f"https://www.youtube.com/watch?v={video_id}"
    opts = _stream_opts()

    raw = await _run_ydl(opts, yt_url)

    # Walk through formats to find the best audio-only stream
    formats = raw.get("formats") or []
    audio_format = _pick_best_audio(formats)

    if not audio_format:
        raise HTTPException(status_code=404, detail="No suitable audio stream found")

    stream_url = audio_format.get("url")
    if not stream_url:
        raise HTTPException(status_code=502, detail="Resolved format has no URL")

    ext = audio_format.get("ext", "webm")
    content_type_map = {
        "webm": "audio/webm",
        "m4a":  "audio/mp4",
        "mp4":  "audio/mp4",
        "ogg":  "audio/ogg",
        "opus": "audio/ogg; codecs=opus",
        "aac":  "audio/aac",
        "mp3":  "audio/mpeg",
    }
    content_type = content_type_map.get(ext, "audio/webm")

    result = {
        "url":           stream_url,
        "ext":           ext,
        "content_type":  content_type,
        "video_id":      video_id,
        "title":         raw.get("title", ""),
        "artist":        raw.get("uploader", raw.get("channel", "")),
        "duration":      raw.get("duration"),
        "thumbnail":     _best_thumbnail(raw.get("thumbnails") or []),
        "abr":           audio_format.get("abr"),
    }

    _url_cache.set(video_id, result)
    return result


def _pick_best_audio(formats: list[dict]) -> Optional[dict]:
    """
    Selects the optimal audio-only format, preferring:
      opus/webm > m4a/aac > any audio
    Sorted by bitrate descending within each tier.
    """
    audio_only = [
        f for f in formats
        if f.get("vcodec") in ("none", None, "") and f.get("acodec") not in ("none", None, "")
        and f.get("url")
    ]

    # Tier 1: opus
    opus = sorted(
        [f for f in audio_only if f.get("acodec", "").startswith("opus")],
        key=lambda f: f.get("abr") or 0, reverse=True
    )
    if opus:
        return opus[0]

    # Tier 2: m4a / aac
    m4a = sorted(
        [f for f in audio_only if f.get("ext") in ("m4a", "aac")],
        key=lambda f: f.get("abr") or 0, reverse=True
    )
    if m4a:
        return m4a[0]

    # Tier 3: any audio-only
    rest = sorted(audio_only, key=lambda f: f.get("abr") or 0, reverse=True)
    if rest:
        return rest[0]

    # Tier 4: any format with audio (may include video)
    any_audio = sorted(
        [f for f in formats if f.get("url") and f.get("acodec") not in ("none", None)],
        key=lambda f: f.get("tbr") or f.get("abr") or 0, reverse=True
    )
    return any_audio[0] if any_audio else None


def _best_thumbnail(thumbnails: list[dict]) -> str:
    """Return the highest-quality thumbnail URL."""
    if not thumbnails:
        return ""
    # Prefer thumbnails with explicit dimensions, pick largest
    with_dims = [t for t in thumbnails if t.get("width") and t.get("height")]
    if with_dims:
        return max(with_dims, key=lambda t: t["width"] * t["height"]).get("url", "")
    return thumbnails[-1].get("url", "")


def _normalise_entry(entry: dict) -> Optional[dict]:
    """Convert a raw yt-dlp entry into a clean track dict."""
    vid_id = entry.get("id") or entry.get("video_id")
    if not vid_id:
        return None

    duration = entry.get("duration")
    if duration and duration > 1800:
        return None  # Skip anything > 30 minutes (likely a mix/podcast)

    thumbnails = entry.get("thumbnails") or []
    thumb = _best_thumbnail(thumbnails) or (
        f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
    )

    return {
        "id":        vid_id,
        "title":     entry.get("title", "Unknown Title"),
        "artist":    entry.get("uploader") or entry.get("channel") or "Unknown Artist",
        "duration":  duration,
        "thumbnail": thumb,
        "url":       f"https://www.youtube.com/watch?v={vid_id}",
        "view_count": entry.get("view_count"),
    }


# ── Async HTTP Streaming Pipeline ─────────────────────────────────────────────

CHUNK_SIZE = 65_536  # 64 KiB per chunk


async def _pipe_audio_stream(
    source_url: str,
    request:    Request,
    headers:    dict,
) -> AsyncGenerator[bytes, None]:
    """
    Opens a persistent async connection to the upstream audio URL and
    yields raw byte chunks, respecting HTTP range requests for seeking.
    Closes cleanly if the client disconnects.
    """
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=10, read=30, write=None, pool=5),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    ) as client:
        try:
            async with client.stream(
                "GET",
                source_url,
                headers=headers,
            ) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes(chunk_size=CHUNK_SIZE):
                    # Abort if the browser tab closed
                    if await request.is_disconnected():
                        log.info("Client disconnected — aborting stream")
                        break
                    yield chunk

        except httpx.HTTPStatusError as exc:
            log.error("Upstream HTTP error %s for %s", exc.response.status_code, source_url)
            raise HTTPException(status_code=502, detail="Upstream audio server error")
        except httpx.RequestError as exc:
            log.error("Upstream connection error: %s", exc)
            raise HTTPException(status_code=504, detail="Could not reach audio server")


# ── Application Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AuraStream backend starting up…")
    yield
    log.info("AuraStream backend shutting down…")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AuraStream API",
    version="1.0.0",
    description="Ad-free YouTube Music reverse-proxy backend",
    lifespan=lifespan,
)


# Simple in-memory listening history for discovery seeds
_listen_history: dict[str, list[float]] = {}


def _record_listen(video_id: str, when_ts: float | None = None) -> None:
    """Track listens by video_id with timestamps to support discovery."""
    if when_ts is None:
        when_ts = time.time()
    _listen_history.setdefault(video_id, []).append(when_ts)


def _best_recent_seed(window_seconds: int = 48 * 3600) -> str | None:
    """Return the most-listened video_id within the last window."""
    cutoff = time.time() - window_seconds
    best_vid: str | None = None
    best_count = -1
    for vid, ts_list in _listen_history.items():
        count = sum(1 for ts in ts_list if ts >= cutoff)
        if count > best_count:
            best_count = count
            best_vid = vid
    return best_vid


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "X-Duration"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AuraStream"}


@app.get("/api/search")
async def api_search(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    type: str = Query("songs", description="songs|albums|artists"),
    limit: int = Query(default=20, ge=1, le=50),
):
    """
    Search YouTube for audio tracks matching the query string.
    Returns a JSON array of track metadata objects.

    GET /api/search?q=tame+impala&limit=20
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        tracks = await search_tracks(q.strip(), max_results=limit)
        # Classification routing is lightweight in this phase; UI controls `type`.
        # We still return tracks (single audio items) but allow future deep album/artist parsing.

    except yt_dlp.utils.DownloadError as exc:
        log.error("yt-dlp search error: %s", exc)
        raise HTTPException(status_code=502, detail="Search extraction failed")

    return JSONResponse(content={"query": q, "results": tracks, "count": len(tracks)})


@app.get("/api/track/{video_id}")
async def api_track_info(video_id: str):
    """
    Fetch full metadata for a single track by YouTube video ID.

    GET /api/track/dQw4w9WgXcQ
    """
    _validate_video_id(video_id)

    cached = _info_cache.get(f"track:{video_id}")
    if cached:
        return JSONResponse(content=cached)

    try:
        info = await resolve_stream_url(video_id)
        _record_listen(video_id)

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Track info error for %s: %s", video_id, exc)
        raise HTTPException(status_code=502, detail="Could not fetch track metadata")

    # Strip the raw stream URL from the public metadata response
    public = {k: v for k, v in info.items() if k != "url"}
    _info_cache.set(f"track:{video_id}", public)

    return JSONResponse(content=public)


@app.get("/api/stream/{video_id}")
async def api_stream(video_id: str, request: Request):
    """
    Resolve and proxy the raw audio byte-stream for a YouTube video ID.
    Supports HTTP Range requests for client-side seeking.

    GET /api/stream/dQw4w9WgXcQ
    Range: bytes=0-
    """
    _validate_video_id(video_id)

    # Step 1 — Resolve the direct audio URL (cached after first call)
    try:
        track = await resolve_stream_url(video_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Stream resolve error for %s: %s", video_id, exc)
        raise HTTPException(status_code=502, detail="Stream URL resolution failed")

    source_url   = track["url"]
    content_type = track["content_type"]
    duration     = track.get("duration")

    # Step 2 — Forward client Range header to upstream (enables seeking)
    upstream_headers = dict(_COMMON_HEADERS)
    range_header = request.headers.get("range")
    if range_header:
        upstream_headers["Range"] = range_header

    # Step 3 — HEAD the upstream to capture Content-Length + Accept-Ranges
    response_headers: dict[str, str] = {
        "Content-Type":              content_type,
        "Accept-Ranges":             "bytes",
        "Cache-Control":             "no-store",
        "X-Content-Type-Options":    "nosniff",
        "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
    }
    if duration:
        response_headers["X-Duration"] = str(duration)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            head = await client.head(source_url, headers=upstream_headers)
            if cl := head.headers.get("content-length"):
                response_headers["Content-Length"] = cl
            if cr := head.headers.get("content-range"):
                response_headers["Content-Range"] = cr
    except Exception:
        pass  # Non-fatal — stream will still work without length

    # Step 4 — Determine HTTP status (206 Partial if Range was requested)
    status_code = 206 if range_header else 200

    # Step 5 — Return the streaming response, piping chunks as they arrive
    return StreamingResponse(
        _pipe_audio_stream(source_url, request, upstream_headers),
        status_code=status_code,
        headers=response_headers,
        media_type=content_type,
    )


@app.get("/api/suggest")
async def api_suggest(q: str = Query(..., min_length=1)):
    """
    Lightweight autocomplete suggestions from YouTube search.
    Returns up to 8 quick results for typeahead UI.

    GET /api/suggest?q=billie
    """
    tracks = await search_tracks(q.strip(), max_results=8)
    suggestions = [{"id": t["id"], "title": t["title"], "artist": t["artist"]} for t in tracks]
    return JSONResponse(content={"suggestions": suggestions})


@app.get("/api/related/{video_id}")
async def api_related(video_id: str, limit: int = Query(default=10, le=25)):
    """
    Fetch related tracks by searching for the track title.
    Lightweight heuristic — uses title-based search as a proxy.

    GET /api/related/dQw4w9WgXcQ
    """
    _validate_video_id(video_id)

    info = _info_cache.get(f"track:{video_id}") or {}
    title = info.get("title", "")
    if not title:
        try:
            raw = await resolve_stream_url(video_id)
            title = raw.get("title", "")
        except Exception:
            raise HTTPException(status_code=404, detail="Track not found")

    query = re.sub(r"\s*\(.*?\)\s*|\s*\[.*?]\s*", "", title).strip()
    tracks = await search_tracks(query, max_results=limit + 5)
    related = [t for t in tracks if t["id"] != video_id][:limit]

    return JSONResponse(content={"video_id": video_id, "related": related})


# ── Helpers ───────────────────────────────────────────────────────────────────

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{11}$")


def _validate_video_id(video_id: str) -> None:
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID format")


# ── Lyrics Engine (lrclib.net) ────────────────────────────────────────────────

_lyrics_cache = LRUCache(capacity=256, ttl=86400)  # cache lyrics 24h


@app.get("/api/lyrics/{video_id}")
async def api_lyrics(video_id: str):
    """
    Fetch time-synced LRC lyrics for a track via lrclib.net.
    Falls back to plain text lyrics if synced unavailable.

    GET /api/lyrics/dQw4w9WgXcQ
    """
    _validate_video_id(video_id)

    cached = _lyrics_cache.get(f"lyrics:{video_id}")
    if cached:
        return JSONResponse(content=cached)

    # Get track metadata so we can search lrclib by title + artist
    info = _info_cache.get(f"track:{video_id}") or {}
    title  = info.get("title",  "")
    artist = info.get("artist", "")
    duration = info.get("duration")

    if not title:
        try:
            raw    = await resolve_stream_url(video_id)
            title  = raw.get("title",  "")
            artist = raw.get("artist", "")
            duration = raw.get("duration")
        except Exception:
            raise HTTPException(status_code=404, detail="Track metadata not found")

    # Clean title — strip featuring, remix tags for better matching
    clean_title  = re.sub(r"\s*(feat\.|ft\.|featuring).*", "", title,  flags=re.IGNORECASE).strip()
    clean_title  = re.sub(r"\s*[\(\[].*?[\)\]]", "", clean_title).strip()
    clean_artist = re.sub(r"\s*,.*", "", artist).strip()  # take first artist only

    result = await _fetch_lrclib(clean_title, clean_artist, duration)

    if not result:
        # Retry with original title in case clean was too aggressive
        result = await _fetch_lrclib(title, artist, duration)

    if not result:
        result = {"synced": False, "lines": [], "plain": "", "source": "none"}

    _lyrics_cache.set(f"lyrics:{video_id}", result)
    return JSONResponse(content=result)


async def _fetch_lrclib(title: str, artist: str, duration: Optional[float]) -> Optional[dict]:
    """
    Query lrclib.net for lyrics. Tries synced first, falls back to plain.
    Returns a dict with:
      - synced: bool
      - lines: list of {time: float (seconds), text: str}
      - plain: str (unsyced fallback)
      - source: str
    """
    if not title:
        return None

    params = {"track_name": title}
    if artist:
        params["artist_name"] = artist
    if duration:
        params["duration"] = int(duration)

    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            resp = await client.get(
                "https://lrclib.net/api/search",
                params=params,
                headers={"User-Agent": "AuraStream/1.0 (https://github.com/aurastream)"},
            )
            if resp.status_code != 200:
                return None

            results = resp.json()
            if not results:
                return None

            # Pick best match — prefer synced lyrics, closest duration
            best = _pick_best_lyrics(results, duration)
            if not best:
                return None

            synced_lrc  = best.get("syncedLyrics")  or ""
            plain_text  = best.get("plainLyrics")   or ""

            if synced_lrc:
                lines = _parse_lrc(synced_lrc)
                return {
                    "synced": True,
                    "lines":  lines,
                    "plain":  plain_text,
                    "source": "lrclib",
                }
            elif plain_text:
                return {
                    "synced": False,
                    "lines":  [],
                    "plain":  plain_text,
                    "source": "lrclib",
                }

    except Exception as exc:
        log.warning("lrclib fetch error: %s", exc)

    return None


def _pick_best_lyrics(results: list, duration: Optional[float]) -> Optional[dict]:
    """
    From lrclib search results, prefer:
    1. Has synced lyrics + duration close to track duration
    2. Has synced lyrics
    3. Has plain lyrics
    """
    synced  = [r for r in results if r.get("syncedLyrics")]
    plain   = [r for r in results if r.get("plainLyrics")]

    def duration_diff(r):
        d = r.get("duration") or 0
        return abs(d - (duration or 0))

    if synced:
        return min(synced, key=duration_diff)
    if plain:
        return min(plain, key=duration_diff)
    return None


def _parse_lrc(lrc: str) -> list[dict]:
    """
    Parse an LRC string into a list of {time: float, text: str} dicts.
    Handles standard [MM:SS.xx] and [MM:SS.xxx] timestamp formats.
    """
    pattern = re.compile(r"\[(\d{1,2}):(\d{2})\.(\d{2,3})\](.*)")
    lines   = []

    for raw_line in lrc.splitlines():
        m = pattern.match(raw_line.strip())
        if not m:
            continue
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        ms_raw  = m.group(3)
        # Normalise to milliseconds
        ms = int(ms_raw) if len(ms_raw) == 3 else int(ms_raw) * 10
        text = m.group(4).strip()

        time_s = minutes * 60 + seconds + ms / 1000.0
        lines.append({"time": round(time_s, 3), "text": text})

    return sorted(lines, key=lambda x: x["time"])


@app.get("/")
async def serve_frontend():45
    from fastapi.responses import FileResponse
    return FileResponse("index.html")