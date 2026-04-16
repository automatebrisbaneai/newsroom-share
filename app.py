import json
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Rate limiter (Phase 0 — per-IP caps via slowapi)
# /clean  : 10/minute + 100/day  — paid OpenRouter proxy, tightest cap
# /submit : 30/hour              — genuine bursts from event day covered
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# Phase 0: hard cap for /submit payload (bytes).  1 GB = 1 073 741 824 bytes.
MAX_SUBMIT_BYTES = 1 * 1024 * 1024 * 1024

# Phase 0: accepted MIME prefixes for uploaded files
ALLOWED_MIME_PREFIXES = ("image/", "video/")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PB_NEWS_URL = os.environ.get("PB_NEWS_URL", "https://pb-news.croquetwade.com")
PB_NEWS_EMAIL = os.environ.get("PB_NEWS_ADMIN_EMAIL", os.environ.get("PB_ADMIN_EMAIL", ""))
PB_NEWS_PASSWORD = os.environ.get("PB_NEWS_ADMIN_PASSWORD", os.environ.get("PB_ADMIN_PASSWORD", ""))
CLEAN_MODEL = "deepseek/deepseek-v3.2"

# ---------------------------------------------------------------------------
# Default timeouts (Phase 2a — httpx per-phase timeouts)
# connect=5s  : TCP handshake must complete quickly
# read=30s    : normal API responses (OpenRouter + PB JSON endpoints)
# write=30s   : request body upload
# pool=5s     : waiting for a free connection from the pool
# For _post_record with file uploads, read timeout is extended to 120s inline.
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
FILE_UPLOAD_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create a shared httpx.AsyncClient for the app lifetime."""
    client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    application.state.http = client
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _load_croquet_dictionary() -> dict:
    """Load croquet-dictionary.json from shared/ — tries app dir first, then parent."""
    for base in (Path(__file__).parent, Path(__file__).parent.parent):
        p = base / "shared" / "croquet-dictionary.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {"terms": [], "players": []}


_DICTIONARY = _load_croquet_dictionary()
_DICTIONARY_HINT = (
    "The following are valid croquet terms that may appear in the transcript — "
    "correct any misheard words to match these exactly: "
    + ", ".join(_DICTIONARY.get("terms", []))
    + ". "
    "The following are Queensland croquet player names — "
    "correct any misheard names to match these exactly: "
    + ", ".join(_DICTIONARY.get("players", []))
    + ". "
) if (_DICTIONARY.get("terms") or _DICTIONARY.get("players")) else ""

_pb_token: str = ""


async def _auth(client: httpx.AsyncClient) -> str:
    r = await client.post(
        f"{PB_NEWS_URL}/api/collections/_superusers/auth-with-password",
        json={"identity": PB_NEWS_EMAIL, "password": PB_NEWS_PASSWORD},
    )
    r.raise_for_status()
    return r.json()["token"]


async def get_token(client: httpx.AsyncClient) -> str:
    global _pb_token
    if not _pb_token:
        _pb_token = await _auth(client)
    return _pb_token


async def refresh_token(client: httpx.AsyncClient) -> str:
    global _pb_token
    _pb_token = await _auth(client)
    return _pb_token


async def _post_record(client: httpx.AsyncClient, token: str, fields: dict, file_parts: list):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{PB_NEWS_URL}/api/collections/news_articles/records"
    if file_parts:
        # Multipart upload — use per-request extended read timeout for large files.
        files_payload = {}
        data_payload = fields
        # httpx multipart: files dict maps field name → (filename, content, mime)
        for idx, (field_name, (filename, content, mime)) in enumerate(file_parts):
            key = f"{field_name}_{idx}" if idx > 0 else field_name
            files_payload[key] = (filename, content, mime)
        return await client.post(
            url,
            headers=headers,
            data=data_payload,
            files=files_payload,
            timeout=FILE_UPLOAD_TIMEOUT,
        )
    return await client.post(
        url,
        headers=headers,
        json=fields,
    )


class TranscriptRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/shared/{filename}")
async def shared_file(filename: str):
    from fastapi.responses import Response
    safe = Path(filename).name  # strip any path traversal
    # Docker: shared/ is inside app dir; local dev: shared/ is sibling at apps/shared/
    file_path = Path(__file__).parent / "shared" / safe
    if not file_path.is_file():
        file_path = Path(__file__).parent.parent / "shared" / safe
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=file_path.read_text(encoding="utf-8"),
        media_type="application/javascript",
    )


@app.post("/clean")
@limiter.limit("10/minute")
@limiter.limit("100/day")
async def clean_transcript(request: Request, req: TranscriptRequest):
    client: httpx.AsyncClient = request.app.state.http
    try:
        res = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": CLEAN_MODEL,
                "messages": [{
                    "role": "user",
                    "content": (
                        _DICTIONARY_HINT
                        + "Clean up this voice transcript into readable, properly punctuated text. "
                        "The input has no punctuation — you must add it. "
                        "Capitalise the start of sentences. Add commas, full stops, and question marks where needed. "
                        "Remove filler words (um, uh, like, you know, sort of). "
                        "Fix run-on sentences by breaking them up. "
                        "Fix speech recognition errors by reading the full sentence for context — "
                        "use whole-sentence inference, not just adjacent words. "
                        "Keep the meaning and tone exactly as intended. "
                        "Return only the cleaned text, nothing else.\n\n"
                        + req.text
                    ),
                }],
                "max_tokens": 4096,
            },
        )
        data = res.json()
        if "choices" not in data:
            print(f"[clean] OpenRouter error response: {data}")
            raise HTTPException(status_code=502, detail="Transcript cleaning failed, please try again.")
        return {"cleaned": data["choices"][0]["message"]["content"]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[clean] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Transcript cleaning failed, please try again.")


@app.post("/submit")
@limiter.limit("30/hour")
async def submit(
    request: Request,
    name: str = Form(...),
    event: str = Form(""),
    caption: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    # Phase 0: reject oversized payloads before reading file bytes into RAM.
    # Check Content-Length header first (fast path — client must declare size).
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_SUBMIT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Payload too large. Maximum total upload size is 1 GB.",
                )
        except ValueError:
            pass  # malformed header — proceed and catch at read time

    name = name.strip()
    event = event.strip()

    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")

    title = f"{event} — from {name}" if event else f"Newsroom submission from {name}"

    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[\s_-]+', '-', slug).strip('-')[:60]
    slug = f"{slug}-{int(time.time())}"

    fields = {
        "title": title,
        "slug": slug,
        "body": caption,
        "author_name": name,
        "category": "Events",
        "status": "submitted",
    }

    # Read all files; enforce MIME whitelist and byte-count cap as we go.
    file_parts: list = []
    cover_set = False
    total_bytes = 0
    for f in files:
        if not f.filename or not f.content_type:
            continue

        # Phase 0: MIME whitelist — accept image/* and video/* only.
        # Note: we trust the client-declared content_type for now.
        # Byte-level sniffing (python-magic) is deferred to Phase 5 — it requires
        # libmagic on the Docker image which adds build complexity.
        if not any(f.content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{f.content_type}'. Only images and videos are accepted.",
            )

        content = await f.read()

        # Phase 0: byte-count safety net — catch any payload that slipped past
        # the Content-Length header check (e.g. chunked transfer).
        total_bytes += len(content)
        if total_bytes > MAX_SUBMIT_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Payload too large. Maximum total upload size is 1 GB.",
            )

        if not cover_set and f.content_type.startswith("image/"):
            file_parts.append(("cover_image", (f.filename, content, f.content_type)))
            cover_set = True
        else:
            file_parts.append(("media", (f.filename, content, f.content_type)))

    client: httpx.AsyncClient = request.app.state.http
    try:
        token = await get_token(client)
        r = await _post_record(client, token, fields, file_parts)
        if r.status_code in (401, 403):
            token = await refresh_token(client)
            r = await _post_record(client, token, fields, file_parts)
        if not r.is_success:
            print(f"[submit] PocketBase error {r.status_code}: {r.text}")
            raise HTTPException(status_code=502, detail="Submission failed, please try again.")
        return {"ok": True, "id": r.json().get("id")}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[submit] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Submission failed, please try again.")
