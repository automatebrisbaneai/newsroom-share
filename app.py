import json
import os
import re
import tempfile
import uuid as uuid_lib
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
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
# Phase 1B: scoped service account — writes to `submissions` only, never `news_articles`.
# Uses the `users` auth collection, NOT `_superusers`.
# The old PB_NEWS_ADMIN_* vars are kept in Coolify for the promote/reject scripts only.
PB_NEWS_SUBMISSIONS_EMAIL = os.environ.get("PB_NEWS_SUBMISSIONS_EMAIL", "")
PB_NEWS_SUBMISSIONS_PASSWORD = os.environ.get("PB_NEWS_SUBMISSIONS_PASSWORD", "")
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
    """Authenticate as the scoped service account (users collection, not _superusers).
    This account can only create records in the submissions collection."""
    r = await client.post(
        f"{PB_NEWS_URL}/api/collections/users/auth-with-password",
        json={"identity": PB_NEWS_SUBMISSIONS_EMAIL, "password": PB_NEWS_SUBMISSIONS_PASSWORD},
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
    """Post to the submissions collection (not news_articles).
    The scoped service account only has createRule on submissions.

    file_parts is a list of (field_name, (filename, file_handle, mime)) tuples.
    file_handle is a seeked-to-0 SpooledTemporaryFile (or any file-like object).
    httpx streams from the handle without loading it all into RAM.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{PB_NEWS_URL}/api/collections/submissions/records"
    if file_parts:
        # Multipart upload — use per-request extended read timeout for large files.
        files_payload = {}
        data_payload = fields
        # httpx multipart: files dict maps field name → (filename, file_handle, mime)
        # httpx reads from the handle in chunks — no full-buffer in memory.
        for idx, (field_name, (filename, file_handle, mime)) in enumerate(file_parts):
            key = f"{field_name}_{idx}" if idx > 0 else field_name
            files_payload[key] = (filename, file_handle, mime)
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
    submission_uuid: Optional[str] = Form(None),
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

    # Phase 2: use client-provided UUID if present; fall back to server-generated.
    # Client UUID enables idempotency: a retry with the same UUID returns the
    # existing record rather than creating a duplicate.
    if not submission_uuid:
        submission_uuid = str(uuid_lib.uuid4())

    title = f"{event} — from {name}" if event else f"Newsroom submission from {name}"

    # Phase 2: UUID suffix instead of timestamp — eliminates concurrent collisions.
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[\s_-]+', '-', slug).strip('-')[:55]
    slug = f"{slug}-{uuid_lib.uuid4().hex[:8]}"

    # Capture client IP and user agent for abuse tracing.
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host
    user_agent = request.headers.get("user-agent", "")[:500]

    fields = {
        "title": title,
        "slug": slug,
        "body": caption,
        "author_name": name,
        "category": "Events",
        "submission_uuid": submission_uuid,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "status": "pending",
    }

    # Phase 2: stream uploads via SpooledTemporaryFile.
    # Files up to 10 MB stay in RAM; larger files spill to disk.
    # We count bytes as we stream so the 1 GB cap is still enforced
    # even for chunked-transfer requests without a Content-Length header.
    SPOOL_MEM = 10 * 1024 * 1024  # 10 MB in-memory threshold per file

    file_parts: list = []
    tempfiles: list = []  # keep refs alive until after httpx finishes
    cover_set = False
    total_bytes = 0

    try:
        for f in files:
            if not f.filename or not f.content_type:
                continue

            # Phase 0: MIME whitelist — accept image/* and video/* only.
            # Note: we trust the client-declared content_type for now.
            # Byte-level sniffing (python-magic) is deferred to Phase 5.
            if not any(f.content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type '{f.content_type}'. Only images and videos are accepted.",
                )

            # Stream into a SpooledTemporaryFile in 64 KB chunks.
            tmp = tempfile.SpooledTemporaryFile(max_size=SPOOL_MEM)
            tempfiles.append(tmp)
            chunk_size = 64 * 1024  # 64 KB
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                # Phase 0: byte-count safety net for chunked transfers.
                if total_bytes > MAX_SUBMIT_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Payload too large. Maximum total upload size is 1 GB.",
                    )
                tmp.write(chunk)

            tmp.seek(0)  # rewind so httpx can read from the start

            if not cover_set and f.content_type.startswith("image/"):
                file_parts.append(("cover_image", (f.filename, tmp, f.content_type)))
                cover_set = True
            else:
                file_parts.append(("media", (f.filename, tmp, f.content_type)))

        http_client: httpx.AsyncClient = request.app.state.http

        # Phase 2: idempotency check — query PB for an existing record with this UUID.
        # If found, return it immediately (handles retries + network failures gracefully).
        try:
            token = await get_token(http_client)
            existing = await http_client.get(
                f"{PB_NEWS_URL}/api/collections/submissions/records",
                params={"filter": f'(submission_uuid="{submission_uuid}")', "perPage": 1},
                headers={"Authorization": f"Bearer {token}"},
            )
            if existing.status_code == 200:
                items = existing.json().get("items", [])
                if items:
                    print(f"[submit] Idempotency hit — returning existing record {items[0]['id']}")
                    return {"ok": True, "id": items[0]["id"]}
        except Exception as e:
            # Non-fatal — if the check fails we proceed to create.
            # The PB unique index is the hard backstop.
            print(f"[submit] Idempotency pre-check error (non-fatal): {e}")

        try:
            token = await get_token(http_client)
            r = await _post_record(http_client, token, fields, file_parts)
            if r.status_code in (401, 403):
                token = await refresh_token(http_client)
                # Rewind all tempfiles before the retry attempt.
                for _, (_, fh, _) in file_parts:
                    fh.seek(0)
                r = await _post_record(http_client, token, fields, file_parts)
            if not r.is_success:
                # PB unique constraint violation on submission_uuid = duplicate submission.
                # Treat as idempotent success: the first attempt got through.
                if r.status_code == 400 and "submission_uuid" in r.text:
                    print(f"[submit] PB unique constraint on submission_uuid — treating as success")
                    return {"ok": True, "id": None}
                print(f"[submit] PocketBase error {r.status_code}: {r.text}")
                raise HTTPException(status_code=502, detail="Submission failed, please try again.")
            return {"ok": True, "id": r.json().get("id")}
        except HTTPException:
            raise
        except Exception as e:
            print(f"[submit] Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="Submission failed, please try again.")
    finally:
        # Always close tempfiles — SpooledTemporaryFile deletes disk spill on close.
        for tmp in tempfiles:
            try:
                tmp.close()
            except Exception:
                pass
