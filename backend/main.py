"""
AuraStream — FastAPI Reverse-Proxy Audio Backend
Phase 2: Segmented Search & Media Categorization Upgrades
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
    """Run yt-dlp extraction in a worker thread to avoid blocking the async loop."""
    def _extract() -> dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url_or_query, download=False)

    return await asyncio.to_thread(_extract)


async def search_tracks(query: str, search_type: str = "songs", max_results: int = 20) -> list[dict]:
    """
    Search YouTube with specific category filters (songs, albums, artists).
    Returns a clean, uniform metadata payload dictionary map matching UI specifications.
    """
    cache_key = f"search:{search_type}:{query}:{max_results}"
    cached = _info_cache.get(cache_key)
    if cached:
        log.info("Cache HIT — search type [%s]: %s", search_type, query)
        return cached["results"]

    # Apply specialized query syntax based on the tab selection
    if search_type == "albums":
        yt_query = f"ytsearch{max_results}:{query} full album music"
    elif search_type == "artists":
        yt_query = f"ytsearch{max_results}:{query} topic artist music"
    else:  # default to "songs"
        yt_query = f"ytsearch{max_results}:{query} audio track music"

    opts = _search_opts()
    raw = await _run_ydl(opts, yt_query)
    entries = raw.get("entries") or []
    results: list[dict] = []

    for entry in entries:
        if not entry:
            continue
        
        # Parse data depending on the expected search tab category format
        if search_type == "albums":
            parsed_item = _normalise_album_entry(entry)
        elif search_type == "artists":
            parsed_item = _normalise_artist_entry(entry)
        else:
            parsed_item = _normalise_entry(entry)
            
        if parsed_item:
            results.append(parsed_item)

    _info_cache.set(cache_key, {"results": results})
    return results


async def resolve_stream_url(video_id: str) -> dict:
    """Resolves the direct audio stream URL for a given YouTube video ID."""
    cached = _url_cache.get(video_id)
    if cached:
        log.info("Cache HIT — stream URL: %s", video_id)
        return cached

    yt_url = f"https://www.youtube.com/watch?v={video_id}"
    opts = _stream_opts()

    raw = await _run_ydl(opts, yt_url)
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
    """Selects the optimal audio-only format, preferring opus/webm over m4a/aac."""
    audio_only = [
        f for f in formats
        if f.get("vcodec") in ("none", None, "") and f.get("acodec") not in ("none", None, "")
        and f.get("url")
    ]

    opus = sorted(
        [f for f in audio_only if f.get("acodec", "").startswith("opus")],
        key=lambda f: f.get("abr") or 0, reverse=True
    )
    if opus:
        return opus[0]

    m4a = sorted(
        [f for f in audio_only if f.get("ext") in ("m4a", "aac")],
        key=lambda f: f.get("abr") or 0, reverse=True
    )
    if m4a:
        return m4a[0]

    rest = sorted(audio_only, key=lambda f: f.get("abr") or 0, reverse=True)
    if rest:
        return rest[0]

    any_audio = sorted(
        [f for f in formats if f.get("url") and f.get("acodec") not in ("none", None)],
        key=lambda f: f.get("tbr") or f.get("abr") or 0, reverse=True
    )
    return any_audio[0] if any_audio else None


def _best_thumbnail(thumbnails: list[dict]) -> str:
    """Return the highest-quality thumbnail URL."""
    if not thumbnails:
        return ""
    with_dims = [t for t in thumbnails if t.get("width") and t.get("height")]
    if with_dims:
        return max(with_dims, key=lambda t: t["width"] * t["height"]).get("url", "")
    return thumbnails[-1].get("url", "")


# ── Categorization Normalizers ────────────────────────────────────────────────

def _normalise_entry(entry: dict) -> Optional[dict]:
    """Convert a raw yt-dlp entry into a clean song/track dict."""
    vid_id = entry.get("id") or entry.get("video_id")
    if not vid_id:
        return None

    duration = entry.get("duration")
    if duration and duration > 1800:
        return None  # Skip anything > 30 minutes for normal track listings

    thumb = _best_thumbnail(entry.get("thumbnails") or []) or f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"

    return {
        "id":        vid_id,
        "title":     entry.get("title", "Unknown Title"),
        "artist":    entry.get("uploader") or entry.get("channel") or "Unknown Artist",
        "duration":  duration,
        "thumbnail": thumb,
        "url":       f"https://www.youtube.com/watch?v={vid_id}",
    }


def _normalise_album_entry(entry: dict) -> Optional[dict]:
    """Sanitize and return raw collection listings matching structural Album cards."""
    vid_id = entry.get("id") or entry.get("video_id")
    if not vid_id:
        return None

    title = entry.get("title", "Unknown Album")
    # Strip messy suffixes typical of video file naming conventions
    title = re.sub(r"\[Full Album\]|\(Full Album\)", "", title, flags=re.IGNORECASE).strip()
    thumb = _best_thumbnail(entry.get("thumbnails") or []) or f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"

    return {
        "id":        vid_id,
        "title":     title,
        "artist":    entry.get("uploader") or entry.get("channel") or "Unknown Creator",
        "duration":  entry.get("duration"),
        "thumbnail": thumb,
        "url":       f"https://www.youtube.com/watch?v={vid_id}",
    }


def _normalise_artist_entry(entry: dict) -> Optional[dict]:
    """Sanitize and return profile structures matching Artist avatar frames."""
    vid_id = entry.get("id") or entry.get("video_id")
    if not vid_id:
        return None

    artist_name = entry.get("uploader") or entry.get("channel") or "Unknown Artist"
    artist_name = re.sub(r" - Topic", "", artist_name, flags=re.IGNORECASE).strip()
    thumb = _best_thumbnail(entry.get("thumbnails") or []) or f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"

    return {
        "id":        vid_id,
        "title":     artist_name,
        "artist":    "Official Profile",
        "duration":  None,
        "thumbnail": thumb,
        "url":       f"https://www.youtube.com/watch?v={vid_id}",
    }


# ── Async HTTP Streaming Pipeline ─────────────────────────────────────────────

CHUNK_SIZE = 65_536  # 64 KiB per chunk


async def _pipe_audio_stream(
    source_url: str,
    request:    Request,
    headers:    dict,
) -> AsyncGenerator[bytes, None]:
    """Opens a persistent async connection to upstream URLs and pipes direct bytes."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=10, read=30, write=None, pool=5),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    ) as client:
        try:
            async with client.stream("GET", source_url, headers=headers) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes(chunk_size=CHUNK_SIZE):
                    if await request.is_disconnected():
                        log.info("Client disconnected — aborting stream proxy loop")
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


# ── FastAPI App Initializer ───────────────────────────────────────────────────

app = FastAPI(
    title="AuraStream API",
    version="2.0.0",
    description="Segmented YouTube Music reverse-proxy backend engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open to facilitate seamless sandboxed environment loads
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "X-Duration"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AuraStream Engine"}


@app.get("/api/search")
async def api_search(
    q: str = Query(..., min_length=1, max_length=200, description="Search keyword query"),
    type: str = Query("songs", description="songs|albums|artists"),
    limit: int = Query(default=20, ge=1, le=50),
):
    """
    Search YouTube with adaptive parameter filters matching frontend UI tabs.
    Returns categorized collection objects.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be blank")

    # Sanitize incoming type requests to avoid system errors
    target_type = type.lower().strip()
    if target_type not in ("songs", "albums", "artists", "all"):
        target_type = "songs"

    try:
        # Resolve 'all' query logic by defaulting to primary songs catalog
        search_filter = "songs" if target_type == "all" else target_type
        tracks = await search_tracks(q.strip(), search_type=search_filter, max_results=limit)

    except yt_dlp.utils.DownloadError as exc:
        log.error("yt-dlp search extraction engine error: %s", exc)
        raise HTTPException(status_code=502, detail="Search extraction layer failed")

    return JSONResponse(content={"query": q, "type": target_type, "results": tracks, "count": len(tracks)})


@app.get("/api/track/{video_id}")
async def api_track_info(video_id: str):
    """Fetch complete structural metadata payload mappings by YouTube video ID."""
    _validate_video_id(video_id)

    cached = _info_cache.get(f"track:{video_id}")
    if cached:
        return JSONResponse(content=cached)

    try:
        info = await resolve_stream_url(video_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Track metadata resolution error for %s: %s", video_id, exc)
        raise HTTPException(status_code=502, detail="Could not resolve track metadata structure")

    public = {k: v for k, v in info.items() if k != "url"}
    _info_cache.set(f"track:{video_id}", public)

    return JSONResponse(content=public)


@app.get("/api/stream/{video_id}")
async def api_stream(video_id: str, request: Request):
    """Proxy direct audio stream segments. Supports HTTP Range query headers for seeking."""
    _validate_video_id(video_id)

    try:
        track = await resolve_stream_url(video_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Stream resolution pipeline crash for %s: %s", video_id, exc)
        raise HTTPException(status_code=502, detail="Stream tracking generation failed")

    source_url   = track["url"]
    content_type = track["content_type"]
    duration     = track.get("duration")

    upstream_headers = dict(_COMMON_HEADERS)
    range_header = request.headers.get("range")
    if range_header:
        upstream_headers["Range"] = range_header

    response_headers: dict[str, str] = {
        "Content-Type":              content_type,
        "Accept-Ranges":             "bytes",
        "Cache-Control":             "no-store",
        "X-Content-Type-Options":    "nosniff",
        "Access-Control-Allow-Origin": "*",
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
        pass

    status_code = 206 if range_header else 200

    return StreamingResponse(
        _pipe_audio_stream(source_url, request, upstream_headers),
        status_code=status_code,
        headers=response_headers,
        media_type=content_type,
    )


@app.get("/api/lyrics/{video_id}")
async def api_lyrics(video_id: str):
    """Fetch time-synced lyric nodes parsed dynamically into millisecond array models."""
    _validate_video_id(video_id)

    cached = _lyrics_cache.get(f"lyrics:{video_id}")
    if cached:
        return JSONResponse(content=cached)

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
            raise HTTPException(status_code=404, detail="Track reference points missing")

    clean_title  = re.sub(r"\s*(feat\.|ft\.|featuring).*", "", title,  flags=re.IGNORECASE).strip()
    clean_title  = re.sub(r"\s*[\(\[].*?[\)\]]", "", clean_title).strip()
    clean_artist = re.sub(r"\s*,.*", "", artist).strip()

    result = await _fetch_lrclib(clean_title, clean_artist, duration)
    if not result:
        result = await _fetch_lrclib(title, artist, duration)
    if not result:
        result = {"synced": False, "lines": [], "plain": "", "source": "none"}

    _lyrics_cache.set(f"lyrics:{video_id}", result)
    return JSONResponse(content=result)


# ── Helpers ───────────────────────────────────────────────────────────────────

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{11}$")


def _validate_video_id(video_id: str) -> None:
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid target media token formatting")


async def _fetch_lrclib(title: str, artist: str, duration: Optional[float]) -> Optional[dict]:
    if not title:
        return None
    params = {"track_name": title}
    if artist:
        params["artist_name"] = artist
    if duration:
        params["duration"] = int(duration)

    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            resp = await client.get("https://lrclib.net/api/search", params=params)
            if resp.status_code != 200:
                return None
            results = resp.json()
            if not results:
                return None

            best = _pick_best_lyrics(results, duration)
            if not best:
                return None

            synced_lrc  = best.get("syncedLyrics")  or ""
            plain_text  = best.get("plainLyrics")   or ""

            if synced_lrc:
                lines = _parse_lrc(synced_lrc)
                return {"synced": True, "lines": lines, "plain": plain_text, "source": "lrclib"}
            elif plain_text:
                return {"synced": False, "lines": [], "plain": plain_text, "source": "lrclib"}
    except Exception:
        pass
    return None


def _pick_best_lyrics(results: list, duration: Optional[float]) -> Optional[dict]:
    synced  = [r for r in results if r.get("syncedLyrics")]
    plain   = [r for r in results if r.get("plainLyrics")]
    def duration_diff(r): return abs((r.get("duration") or 0) - (duration or 0))
    if synced: return min(synced, key=duration_diff)
    if plain: return min(plain, key=duration_diff)
    return None


def _parse_lrc(lrc: str) -> list[dict]:
    pattern = re.compile(r"\[(\d{1,2}):(\d{2})\.(\d{2,3})\](.*)")
    lines   = []
    for raw_line in lrc.splitlines():
        m = pattern.match(raw_line.strip())
        if not m: continue
        minutes, seconds, ms_raw = int(m.group(1)), int(m.group(2)), m.group(3)
        ms = int(ms_raw) if len(ms_raw) == 3 else int(ms_raw) * 10
        time_s = minutes * 60 + seconds + ms / 1000.0
        lines.append({"time": round(time_s, 3), "text": m.group(4).strip()})
    return sorted(lines, key=lambda x: x["time"])

_lyrics_cache = LRUCache(capacity=256, ttl=86400)