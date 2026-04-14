import os
import re
import time
import requests as http_requests
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PB_NEWS_URL = os.environ.get("PB_NEWS_URL", "https://pb-news.croquetwade.com")
PB_NEWS_EMAIL = os.environ.get("PB_NEWS_ADMIN_EMAIL", os.environ.get("PB_ADMIN_EMAIL", ""))
PB_NEWS_PASSWORD = os.environ.get("PB_NEWS_ADMIN_PASSWORD", os.environ.get("PB_ADMIN_PASSWORD", ""))
CLEAN_MODEL = "deepseek/deepseek-v3.2"

_pb_token: str = ""


def _auth() -> str:
    r = http_requests.post(
        f"{PB_NEWS_URL}/api/collections/_superusers/auth-with-password",
        json={"identity": PB_NEWS_EMAIL, "password": PB_NEWS_PASSWORD},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["token"]


def get_token() -> str:
    global _pb_token
    if not _pb_token:
        _pb_token = _auth()
    return _pb_token


def refresh_token() -> str:
    global _pb_token
    _pb_token = _auth()
    return _pb_token


def _post_record(token: str, fields: dict, file_parts: list):
    if file_parts:
        return http_requests.post(
            f"{PB_NEWS_URL}/api/collections/news_articles/records",
            headers={"Authorization": f"Bearer {token}"},
            data=fields,
            files=file_parts,
            timeout=120,
        )
    return http_requests.post(
        f"{PB_NEWS_URL}/api/collections/news_articles/records",
        headers={"Authorization": f"Bearer {token}"},
        json=fields,
        timeout=30,
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
async def clean_transcript(req: TranscriptRequest):
    res = http_requests.post(
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
                    "Clean up this voice transcript into readable, properly punctuated text. "
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
        timeout=30,
    )
    data = res.json()
    if "choices" not in data:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {data}")
    return {"cleaned": data["choices"][0]["message"]["content"]}


@app.post("/submit")
async def submit(
    name: str = Form(...),
    event: str = Form(""),
    caption: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
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

    # Read all files, assign first image as cover, rest go to media[]
    file_parts: list = []
    cover_set = False
    for f in files:
        if not f.filename or not f.content_type:
            continue
        content = await f.read()
        if not cover_set and f.content_type.startswith("image/"):
            file_parts.append(("cover_image", (f.filename, content, f.content_type)))
            cover_set = True
        else:
            file_parts.append(("media", (f.filename, content, f.content_type)))

    try:
        token = get_token()
        r = _post_record(token, fields, file_parts)
        if r.status_code == 401:
            token = refresh_token()
            r = _post_record(token, fields, file_parts)
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"PocketBase error: {r.text}")
        return {"ok": True, "id": r.json().get("id")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
