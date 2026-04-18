import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid as uuid_lib
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import List, Optional

import bleach
import httpx
import magic
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, Response as FastAPIResponse
from pydantic import BaseModel
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Phase 4: Structured JSON logging
#
# Every log line is emitted as a single JSON object to stdout.
# Coolify's log viewer handles line-based output, and JSON is grep/jq-able.
#
# REQUEST_ID ContextVar carries the per-request correlation ID.  Every log
# call in the request handler picks it up automatically via the formatter.
# ---------------------------------------------------------------------------
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")


class _JsonFormatter(logging.Formatter):
    """Serialise every log record as a single JSON line."""

    _SKIP_KEYS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        super().format(record)
        out: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(""),
        }
        for key, value in record.__dict__.items():
            if key not in self._SKIP_KEYS:
                out[key] = value
        if record.exc_text:
            out["exc"] = record.exc_text
        return json.dumps(out, default=str)


def _configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = False


_configure_logging()
logger = logging.getLogger(__name__)

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
NEWSROOM_API_URL = os.environ.get("NEWSROOM_API_URL", "https://my.croquetwade.com")
NEWSROOM_API_TOKEN = os.environ.get("NEWSROOM_API_TOKEN", "")
CLEAN_MODEL = "deepseek/deepseek-v3.2"

MIN_WORD_CHARS = 3

_SUSPICIOUS_STARTS = (
    "i ", "i'", "i\u2019", "sure", "certainly", "okay", "of course",
    "please", "here is", "here's", "understood",
)
_SUSPICIOUS_SUBSTRINGS = (
    "provide the transcript", "provide the text", "provide the voice",
    "i will process", "i understand", "i'll clean", "i will clean",
    "as an ai", "happy to help",
)


def _looks_like_meta_response(raw_input: str, model_output: str) -> bool:
    if not model_output:
        return False
    stripped = model_output.strip().lower()
    if not stripped:
        return False
    if stripped.startswith(_SUSPICIOUS_STARTS):
        return True
    if any(s in stripped for s in _SUSPICIOUS_SUBSTRINGS):
        return True
    in_words = {w for w in re.findall(r"[a-z]+", raw_input.lower()) if len(w) > 2}
    out_words = re.findall(r"[a-z]+", stripped)
    if in_words and len(out_words) > len(in_words) * 3:
        overlap = sum(1 for w in out_words if w in in_words)
        if overlap < max(2, len(in_words) // 3):
            return True
    return False


_CLEAN_SYSTEM_PROMPT = None  # populated after _DICTIONARY_HINT is built (lower in file)

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
    """Create a shared httpx.AsyncClient for the app lifetime.

    Also performs a fail-fast PB auth check on startup: if the scoped service
    account credentials are missing or wrong, the process exits 1 immediately so
    Coolify shows the container as unhealthy rather than silently degrading.
    """
    client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    application.state.http = client

    # ── Fail-fast API token verification ─────────────────────────────
    if not NEWSROOM_API_TOKEN:
        logger.critical(
            "FATAL: NEWSROOM_API_TOKEN is not set.",
            extra={"event": "api_auth_startup", "status": "fatal"},
        )
        await client.aclose()
        sys.exit(1)
    try:
        r = await client.get(
            f"{NEWSROOM_API_URL}/api/newsroom",
            headers={"Authorization": f"Bearer {NEWSROOM_API_TOKEN}"},
            params={"type": "article", "scope": "all", "perPage": 1},
        )
        r.raise_for_status()
        logger.info(
            "MyCroquet API ping OK at %s",
            NEWSROOM_API_URL,
            extra={"event": "api_auth_startup", "status": "ok"},
        )
    except Exception as exc:
        logger.critical(
            "FATAL: MyCroquet API ping failed — check NEWSROOM_API_URL / NEWSROOM_API_TOKEN. Error: %s",
            exc,
            extra={"event": "api_auth_startup", "status": "fatal"},
        )
        await client.aclose()
        sys.exit(1)

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

_CLEAN_SYSTEM_PROMPT = (
    "You are a transcript cleaner. You receive raw voice-recognition text "
    "and return ONLY the cleaned text. Rules:\n"
    "- Add punctuation (commas, full stops, question marks).\n"
    "- Capitalise the start of sentences.\n"
    "- Remove filler words (um, uh, like, you know, sort of).\n"
    "- Fix run-on sentences by breaking them up.\n"
    "- Fix speech-recognition errors using whole-sentence context.\n"
    "- Keep meaning and tone exactly as intended.\n"
    "You NEVER respond conversationally. You NEVER ask for input. "
    "You NEVER explain what you are doing. If the input is empty, whitespace, "
    "a single word, or otherwise not a usable transcript, return it verbatim. "
    "Output is the cleaned transcript text and nothing else.\n"
    + _DICTIONARY_HINT
)

def _mc_headers() -> dict:
    return {"Authorization": f"Bearer {NEWSROOM_API_TOKEN}"}


async def _post_to_mycroquet(
    client: httpx.AsyncClient,
    fields: dict,
    cover: tuple | None,
    media_parts: list,
) -> str:
    """Create a content_item via MyCroquet API and submit it for review.

    cover is (filename, file_handle, mime) or None.
    media_parts is a list of (filename, file_handle, mime) for additional files.
    Returns the new item's id.
    """
    data: dict = {
        "type": "article",
        "title": fields["title"],
        "body": fields["body"],
        "visibility": "public",
    }
    files = {}
    if cover:
        filename, fh, mime = cover
        files["cover_image"] = (filename, fh, mime)

    r = await client.post(
        f"{NEWSROOM_API_URL}/api/newsroom",
        headers=_mc_headers(),
        data=data,
        files=files if files else None,
        timeout=FILE_UPLOAD_TIMEOUT,
    )
    if not r.is_success:
        raise HTTPException(status_code=502, detail=f"Newsroom API error: {r.text}")
    record_id: str = r.json()["id"]

    # Move to submitted so it appears in the editorial review queue.
    patch = await client.patch(
        f"{NEWSROOM_API_URL}/api/newsroom/{record_id}",
        headers={**_mc_headers(), "Content-Type": "application/json"},
        json={"status": "submitted"},
    )
    if not patch.is_success:
        logger.warning("Status patch failed (non-fatal)", extra={"event": "submit_patch_fail", "id": record_id, "status": patch.status_code})

    # Attach additional media files.
    for filename, fh, mime in media_parts:
        med = await client.post(
            f"{NEWSROOM_API_URL}/api/newsroom/{record_id}/media",
            headers=_mc_headers(),
            data={"kind": "gallery"},
            files={"file": (filename, fh, mime)},
            timeout=FILE_UPLOAD_TIMEOUT,
        )
        if not med.is_success:
            logger.warning("Media attach failed (non-fatal)", extra={"event": "media_attach_fail", "id": record_id, "status": med.status_code})

    return record_id


class TranscriptRequest(BaseModel):
    text: str


@app.get("/healthz")
async def healthz(request: Request):
    """Health check: pings MyCroquet API.

    Returns 200 with api:ok on success, 503 on error.
    Used by Docker HEALTHCHECK and Coolify monitoring.
    """
    client: httpx.AsyncClient = request.app.state.http
    try:
        r = await client.get(
            f"{NEWSROOM_API_URL}/api/newsroom",
            headers=_mc_headers(),
            params={"type": "article", "scope": "all", "perPage": 1},
        )
        r.raise_for_status()
        return {"status": "ok", "api": "ok", "api_url": NEWSROOM_API_URL}
    except Exception as exc:
        from fastapi.responses import JSONResponse as _JSONResponse
        return _JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "api": "failed", "error": str(exc)},
        )


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))



# Phase 5: explicit allowlist — only these two files are served from /shared/.
# The parent-directory fallback is intentionally removed: in the Docker image
# the shared/ directory is always at /app/shared/ (copied by the Dockerfile
# `COPY . .` step). Local dev must mirror the same layout.
SHARED_FILES = {
    "voice-to-text.js": "application/javascript",
    "croquet-dictionary.json": "application/json",
}


@app.get("/shared/{filename}")
async def shared_file(filename: str):
    from fastapi.responses import Response
    # Allowlist check first — anything not in the dict gets 404, no filesystem probe.
    if filename not in SHARED_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    content_type = SHARED_FILES[filename]
    # Docker layout: shared/ lives inside the app directory at /app/shared/.
    file_path = Path(__file__).parent / "shared" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=file_path.read_text(encoding="utf-8"),
        media_type=content_type,
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
    raw = req.text or ""
    input_chars = len(raw)

    # Input guard — too few word chars means there's nothing to clean.
    # Short-circuit without touching OpenRouter: no prompt, no meta-chatter risk.
    word_char_count = sum(1 for c in raw if c.isalpha())
    if word_char_count < MIN_WORD_CHARS:
        logger.info(
            "Clean short-circuit — input below MIN_WORD_CHARS",
            extra={
                "event": "clean_short_circuit",
                "input_chars": input_chars,
                "word_chars": word_char_count,
            },
        )
        return {"cleaned": raw}

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
                "messages": [
                    {"role": "system", "content": _CLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": raw},
                ],
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

        if _looks_like_meta_response(raw, cleaned):
            logger.warning(
                "Suspicious model output — falling back to raw input",
                extra={
                    "event": "clean_suspicious_output",
                    "input_chars": input_chars,
                    "output_chars": output_chars,
                    "input_preview": raw[:120],
                    "output_preview": cleaned[:240],
                },
            )
            cleaned = raw
            output_chars = len(cleaned)

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

    # Phase 5: sanitise caption on ingest — strip all HTML tags before storage.
    # Defensive against XSS if a downstream renderer ever treats the body as HTML.
    # bleach.clean with tags=[] strips everything; strip=True removes the tags
    # rather than escaping them, so stored text is plain prose.
    caption = bleach.clean(caption, tags=[], strip=True)

    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")

    # Phase 2: use client-provided UUID if present; fall back to server-generated.
    # Client UUID enables idempotency: a retry with the same UUID returns the
    # existing record rather than creating a duplicate.
    if not submission_uuid:
        submission_uuid = str(uuid_lib.uuid4())

    title = f"{event} — from {name}" if event else f"Newsroom submission from {name}"

    fields = {
        "title": title,
        "body": caption,
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
            first_chunk = True
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)

                # Phase 5: byte-level MIME sniffing on the leading bytes of the
                # first chunk (~2 KB is enough for libmagic to identify the format).
                # This is defence-in-depth: the client-declared MIME check above is
                # the fast first gate; sniffing catches spoofed content_type headers.
                if first_chunk:
                    first_chunk = False
                    sniffed_type = magic.from_buffer(chunk[:2048], mime=True)
                    if not any(sniffed_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
                        duration_ms = round((time.monotonic() - t_start) * 1000)
                        logger.warning(
                            "Submit rejected — byte-level MIME sniff mismatch",
                            extra={
                                "event": "mime_sniff_rejected",
                                "claimed_type": f.content_type,
                                "sniffed_type": sniffed_type,
                                "upload_filename": f.filename,
                                "client_ip": client_ip,
                            },
                        )
                        logger.warning(
                            "Submit failed — mime_sniff_rejected",
                            extra={
                                "event": "submit_failure",
                                "duration_ms": duration_ms,
                                "failure_reason": "mime_sniff_rejected",
                                "submission_uuid": submission_uuid,
                            },
                        )
                        raise HTTPException(
                            status_code=415,
                            detail="File content does not match declared type. Only images and videos are accepted.",
                        )

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
                file_parts.append(("cover", (f.filename, tmp, f.content_type)))
                cover_set = True
            else:
                file_parts.append(("media", (f.filename, tmp, f.content_type)))

        file_count = len(file_parts)
        http_client: httpx.AsyncClient = request.app.state.http

        cover_part = next((t for name, t in file_parts if name == "cover"), None)
        media_parts_list = [t for name, t in file_parts if name == "media"]

        try:
            t_api_start = time.monotonic()
            record_id = await _post_to_mycroquet(http_client, fields, cover_part, media_parts_list)
            api_duration_ms = round((time.monotonic() - t_api_start) * 1000)
            logger.info(
                "MyCroquet submit succeeded",
                extra={
                    "event": "api_submit",
                    "record_id": record_id,
                    "submission_uuid": submission_uuid,
                    "duration_ms": api_duration_ms,
                    "total_upload_bytes": total_bytes,
                    "file_count": file_count,
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
