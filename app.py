import json
import os
import re
import tempfile
import time
import uuid as uuid_lib
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, Response as FastAPIResponse
from pydantic import BaseModel
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from logging_config import configure_logging, get_logger, REQUEST_ID

# ---------------------------------------------------------------------------
# Logging — configure before first use so startup messages are structured too.
# ---------------------------------------------------------------------------
configure_logging()
logger = get_logger(__name__)

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
    logger.info("Application startup", extra={"event": "app_startup"})
    try:
        yield
    finally:
        await client.aclose()
        logger.info("Application shutdown", extra={"event": "app_shutdown"})


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Phase 4: Request correlation ID middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Assign a correlation ID to every request.

    - Honour X-Request-ID if a reverse proxy has already set one.
    - Otherwise generate a short UUID (12 hex chars — readable in logs).
    - Store in REQUEST_ID ContextVar so every log line in this request picks
      it up automatically.
    - Return the ID in the X-Request-ID response header so clients/upstreams
      can correlate their own logs.
    """
    request_id = request.headers.get("x-request-id") or uuid_lib.uuid4().hex[:12]
    token = REQUEST_ID.set(request_id)

    path = request.url.path
    method = request.method
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    t_start = time.monotonic()
    logger.info(
        "Request start",
        extra={
            "event": "request_start",
            "path": path,
            "method": method,
            "client_ip": client_ip,
        },
    )

    try:
        response = await call_next(request)
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.info(
            "Request end",
            extra={
                "event": "request_end",
                "path": path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.error(
            "Request error",
            extra={
                "event": "request_end",
                "path": path,
                "status": 500,
                "duration_ms": duration_ms,
            },
        )
        raise
    finally:
        REQUEST_ID.reset(token)


# ---------------------------------------------------------------------------
# Phase 4: Global exception handler — structured ERROR log before 500 reply
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        extra={
            "event": "exception",
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
            "path": request.url.path,
        },
        exc_info=True,
    )
    return FastAPIResponse(
        content=json.dumps({"detail": "An unexpected error occurred."}),
        status_code=500,
        media_type="application/json",
    )


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
        logger.info(
            "PocketBase auth — obtaining token",
            extra={"event": "pb_auth_refresh", "reason": "no_token"},
        )
        _pb_token = await _auth(client)
    return _pb_token


async def refresh_token(client: httpx.AsyncClient) -> str:
    global _pb_token
    logger.info(
        "PocketBase auth — refreshing expired token",
        extra={"event": "pb_auth_refresh", "reason": "token_expired_or_rejected"},
    )
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
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    t_start = time.monotonic()
    input_chars = len(req.text)

    try:
        t_or_start = time.monotonic()
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
        or_duration_ms = round((time.monotonic() - t_or_start) * 1000)
        data = res.json()

        if "choices" not in data:
            logger.error(
                "OpenRouter returned error response",
                extra={
                    "event": "openrouter_clean",
                    "status_code": res.status_code,
                    "duration_ms": or_duration_ms,
                    "input_chars": input_chars,
                    "output_chars": 0,
                    "error": str(data),
                    "client_ip": client_ip,
                },
            )
            duration_ms = round((time.monotonic() - t_start) * 1000)
            logger.warning(
                "Clean failed — openrouter_error",
                extra={
                    "event": "clean_failure",
                    "duration_ms": duration_ms,
                    "failure_reason": "openrouter_error",
                },
            )
            raise HTTPException(status_code=502, detail="Transcript cleaning failed, please try again.")

        cleaned = data["choices"][0]["message"]["content"]
        output_chars = len(cleaned)
        or_duration_ms = round((time.monotonic() - t_or_start) * 1000)

        logger.info(
            "OpenRouter clean succeeded",
            extra={
                "event": "openrouter_clean",
                "status_code": res.status_code,
                "duration_ms": or_duration_ms,
                "input_chars": input_chars,
                "output_chars": output_chars,
            },
        )
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.info(
            "Clean succeeded",
            extra={
                "event": "clean_success",
                "duration_ms": duration_ms,
                "input_chars": input_chars,
                "output_chars": output_chars,
            },
        )
        return {"cleaned": cleaned}

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.error(
            "Clean unexpected error",
            extra={
                "event": "clean_failure",
                "duration_ms": duration_ms,
                "failure_reason": "internal_error",
                "exc_type": type(e).__name__,
                "exc_msg": str(e),
            },
        )
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
    t_start = time.monotonic()
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    # Phase 0: reject oversized payloads before reading file bytes into RAM.
    # Check Content-Length header first (fast path — client must declare size).
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            cl_int = int(content_length)
            if cl_int > MAX_SUBMIT_BYTES:
                duration_ms = round((time.monotonic() - t_start) * 1000)
                logger.warning(
                    "Submit rejected — payload too large (Content-Length check)",
                    extra={
                        "event": "size_cap_rejected",
                        "content_length": cl_int,
                        "client_ip": client_ip,
                    },
                )
                logger.warning(
                    "Submit failed — size_cap",
                    extra={
                        "event": "submit_failure",
                        "duration_ms": duration_ms,
                        "failure_reason": "size_cap",
                        "submission_uuid": submission_uuid,
                    },
                )
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
                duration_ms = round((time.monotonic() - t_start) * 1000)
                logger.warning(
                    "Submit rejected — unsupported MIME type",
                    extra={
                        "event": "mime_rejected",
                        "content_type": f.content_type,
                        "upload_filename": f.filename,
                        "client_ip": client_ip,
                    },
                )
                logger.warning(
                    "Submit failed — mime_rejected",
                    extra={
                        "event": "submit_failure",
                        "duration_ms": duration_ms,
                        "failure_reason": "mime_rejected",
                        "submission_uuid": submission_uuid,
                    },
                )
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
                    duration_ms = round((time.monotonic() - t_start) * 1000)
                    logger.warning(
                        "Submit rejected — payload too large (byte-count check)",
                        extra={
                            "event": "size_cap_rejected",
                            "content_length": total_bytes,
                            "client_ip": client_ip,
                        },
                    )
                    logger.warning(
                        "Submit failed — size_cap",
                        extra={
                            "event": "submit_failure",
                            "duration_ms": duration_ms,
                            "failure_reason": "size_cap",
                            "submission_uuid": submission_uuid,
                        },
                    )
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

        file_count = len(file_parts)
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
                    logger.info(
                        "Idempotency hit — returning existing record",
                        extra={
                            "event": "submit_idempotent_hit",
                            "submission_uuid": submission_uuid,
                            "record_id": items[0]["id"],
                        },
                    )
                    return {"ok": True, "id": items[0]["id"]}
        except Exception as e:
            # Non-fatal — if the check fails we proceed to create.
            # The PB unique index is the hard backstop.
            logger.warning(
                "Idempotency pre-check error (non-fatal)",
                extra={
                    "event": "submit_idempotency_check_error",
                    "submission_uuid": submission_uuid,
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e),
                },
            )

        try:
            token = await get_token(http_client)
            t_pb_start = time.monotonic()
            r = await _post_record(http_client, token, fields, file_parts)
            if r.status_code in (401, 403):
                token = await refresh_token(http_client)
                # Rewind all tempfiles before the retry attempt.
                for _, (_, fh, _) in file_parts:
                    fh.seek(0)
                r = await _post_record(http_client, token, fields, file_parts)
            pb_duration_ms = round((time.monotonic() - t_pb_start) * 1000)

            if not r.is_success:
                # PB unique constraint violation on submission_uuid = duplicate submission.
                # Treat as idempotent success: the first attempt got through.
                if r.status_code == 400 and "submission_uuid" in r.text:
                    logger.info(
                        "PB unique constraint on submission_uuid — treating as success",
                        extra={
                            "event": "pb_submit",
                            "status_code": r.status_code,
                            "submission_uuid": submission_uuid,
                            "duration_ms": pb_duration_ms,
                            "total_upload_bytes": total_bytes,
                            "idempotent_constraint": True,
                        },
                    )
                    return {"ok": True, "id": None}

                logger.error(
                    "PocketBase error on submit",
                    extra={
                        "event": "pb_submit",
                        "status_code": r.status_code,
                        "submission_uuid": submission_uuid,
                        "duration_ms": pb_duration_ms,
                        "total_upload_bytes": total_bytes,
                    },
                )
                duration_ms = round((time.monotonic() - t_start) * 1000)
                logger.error(
                    "Submit failed — pb_error",
                    extra={
                        "event": "submit_failure",
                        "duration_ms": duration_ms,
                        "failure_reason": "pb_error",
                        "submission_uuid": submission_uuid,
                    },
                )
                raise HTTPException(status_code=502, detail="Submission failed, please try again.")

            record_id = r.json().get("id")
            logger.info(
                "PocketBase submit succeeded",
                extra={
                    "event": "pb_submit",
                    "status_code": r.status_code,
                    "submission_uuid": submission_uuid,
                    "duration_ms": pb_duration_ms,
                    "total_upload_bytes": total_bytes,
                },
            )

            duration_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                "Submit succeeded",
                extra={
                    "event": "submit_success",
                    "duration_ms": duration_ms,
                    "total_upload_bytes": total_bytes,
                    "file_count": file_count,
                    "submission_uuid": submission_uuid,
                    "rate_limit_remaining": None,  # slowapi doesn't expose remaining easily
                },
            )
            return {"ok": True, "id": record_id}

        except HTTPException:
            raise
        except Exception as e:
            duration_ms = round((time.monotonic() - t_start) * 1000)
            logger.error(
                "Submit unexpected error",
                extra={
                    "event": "submit_failure",
                    "duration_ms": duration_ms,
                    "failure_reason": "internal_error",
                    "submission_uuid": submission_uuid,
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e),
                },
            )
            raise HTTPException(status_code=500, detail="Submission failed, please try again.")
    finally:
        # Always close tempfiles — SpooledTemporaryFile deletes disk spill on close.
        for tmp in tempfiles:
            try:
                tmp.close()
            except Exception:
                pass
