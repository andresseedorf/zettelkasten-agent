import os
import io
import json
import math
import time
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

# Active Gemini 3 fallback chain to handle temporary 503 high-demand spikes
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-pro",
]


def get_clients():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    folder_id = os.environ.get("DRIVE_FOLDER_ID", "").strip()
    raw_sa = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in Vercel Environment Variables.")
    if not folder_id:
        raise RuntimeError("Missing DRIVE_FOLDER_ID in Vercel Environment Variables.")
    if not raw_sa:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Vercel Environment Variables.")

    creds_info = json.loads(raw_sa, strict=False)
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive = build("drive", "v3", credentials=creds)
    ai = genai.Client(api_key=api_key)
    return ai, drive, folder_id, creds_info.get("client_email", "")


class EdgeProposal(BaseModel):
    target_id: str
    relation: Literal["supports", "contradicts", "extends", "analogous_to"]
    reason: str = Field(description="One concise sentence explaining the conceptual link.")


class ZettelSynthesis(BaseModel):
    title: str = Field(description="Short, declarative concept title (3-7 words).")
    permanent_note: str = Field(description="Atomic idea rewritten in clear, standalone prose.")
    tags: List[str] = Field(description="2-4 lowercase topical tags without #.")
    connections: List[EdgeProposal] = Field(
        default_factory=list,
        description="Meaningful conceptual links to candidate notes. Leave empty if unrelated."
    )


class NoteInput(BaseModel):
    quote: str = ""
    thought: str
    source: str = "Unattributed"


def verify_auth(x_passcode: str = Header(default="")):
    expected = os.environ.get("APP_PASSWORD", "").strip()
    if expected and x_passcode.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid passcode.")


def find_drive_file(drive, folder_id: str, filename: str):
    q = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
    res = drive.files().list(
        q=q,
        fields="files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", [])
    return files[0] if files else None


def load_graph(drive, folder_id: str) -> dict:
    file_meta = find_drive_file(drive, folder_id, "graph.json")
    if not file_meta:
        return {"nodes": [], "links": []}

    file_id = file_meta["id"]
    mime_type = file_meta.get("mimeType", "")

    if mime_type == "application/vnd.google-apps.document":
        request = drive.files().export_media(fileId=file_id, mimeType="text/plain")
    else:
        request = drive.files().get_media(fileId=file_id)

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    content = buf.getvalue().decode("utf-8-sig").strip()
    if not content:
        return {"nodes": [], "links": []}

    data = json.loads(content)
    while isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        return {"nodes": [], "links": []}

    data.setdefault("nodes", [])
    data.setdefault("links", [])
    return data


def upload_text_to_drive(drive, folder_id: str, filename: str, content: str, mime_type: str):
    existing = find_drive_file(drive, folder_id, filename)
    if existing:
        upload_mime = (
            "text/plain"
            if existing.get("mimeType") == "application/vnd.google-apps.document"
            else mime_type
        )
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")), mimetype=upload_mime, resumable=False
        )
        drive.files().update(
            fileId=existing["id"], media_body=media, supportsAllDrives=True
        ).execute()
    else:
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")), mimetype=mime_type, resumable=False
        )
        meta = {"name": filename, "parents": [folder_id]}
        drive.files().create(
            body=meta, media_body=media, fields="id", supportsAllDrives=True
        ).execute()


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / denom if denom else 0.0


def parse_synthesis(response) -> ZettelSynthesis:
    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, ZettelSynthesis):
            return parsed
        if isinstance(parsed, dict):
            return ZettelSynthesis.model_validate(parsed)

    raw = (getattr(response, "text", None) or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty response.")

    obj = json.loads(raw)
    while isinstance(obj, str):
        obj = json.loads(obj)
    return ZettelSynthesis.model_validate(obj)


def generate_with_fallback(ai, prompt: str) -> tuple[ZettelSynthesis, str]:
    last_err = None
    for model_name in FALLBACK_MODELS:
        try:
            response = ai.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ZettelSynthesis,
                    temperature=0.2,
                    http_options=types.HttpOptions(timeout=7000),  # 7s cap so Vercel never 504s
                ),
            )
            return parse_synthesis(response), model_name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Model call failed: {last_err}")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health_check():
    try:
        _, drive, folder_id, sa_email = get_clients()
        folder = drive.files().get(fileId=folder_id, fields="id, name, capabilities").execute()
        return {
            "status": "ok",
            "service_account": sa_email,
            "drive_folder_name": folder.get("name"),
            "can_edit_folder": folder.get("capabilities", {}).get("canAddChildren"),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.api_route("/api/verify", methods=["GET", "POST"])
def check_passcode(x_passcode: str = Header(default="")):
    verify_auth(x_passcode)
    return {"status": "unlocked"}


@app.get("/api/graph")
def read_graph(x_passcode: str = Header(default="")):
    verify_auth(x_passcode)
    try:
        _, drive, folder_id, _ = get_clients()
        graph = load_graph(drive, folder_id)
        clean_nodes = [
            {k: v for k, v in n.items() if k != "embedding"}
            for n in graph["nodes"]
            if isinstance(n, dict)
        ]
        return {"nodes": clean_nodes, "links": graph["links"]}
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        line_info = f" (line {tb[-1].lineno})" if tb else ""
        raise HTTPException(status_code=500, detail=f"{e}{line_info}")


@app.post("/api/notes")
def add_note(payload: NoteInput, x_passcode: str = Header(default="")):
    verify_auth(x_passcode)
    try:
        ai, drive, folder_id, _ = get_clients()
        graph = load_graph(drive, folder_id)
        raw_text = f"Quote: {payload.quote}\nSource: {payload.source}\nThought: {payload.thought}"

        emb_res = ai.models.embed_content(model="gemini-embedding-001", contents=raw_text)
        new_emb = list(emb_res.embeddings[0].values)

        scored = [
            (cosine_sim(new_emb, n["embedding"]), n)
            for n in graph["nodes"]
            if isinstance(n, dict) and "embedding" in n
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [
            {"id": n["id"], "title": n["title"], "permanent_note": n["permanent_note"]}
            for _, n in scored[:6]
        ]

        candidates_block = (
            json.dumps(candidates, indent=2)
            if candidates
            else "None yet (this is the first note in the Zettelkasten, so return an empty connections list)."
        )

        prompt = f"""You are a Zettelkasten research assistant.
Distill the user's input into ONE atomic Permanent Note, and evaluate whether it meaningfully connects to any of the candidate existing notes.

NEW INPUT:
{raw_text}

CANDIDATE EXISTING NOTES:
{candidates_block}
"""
        synthesis, used_model = generate_with_fallback(ai, prompt)

        note_id = uuid.uuid4().hex[:6]
        new_node = {
            "id": note_id,
            "title": synthesis.title,
            "permanent_note": synthesis.permanent_note,
            "quote": payload.quote,
            "source": payload.source,
            "tags": synthesis.tags,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "embedding": new_emb,
        }
        graph["nodes"].append(new_node)

        valid_ids = {n["id"] for n in graph["nodes"] if isinstance(n, dict)}
        new_links = []
        for conn in synthesis.connections:
            if conn.target_id in valid_ids and conn.target_id != note_id:
                edge = {
                    "source": note_id,
                    "target": conn.target_id,
                    "relation": conn.relation,
                    "reason": conn.reason,
                }
                graph["links"].append(edge)
                new_links.append(edge)

        upload_text_to_drive(drive, folder_id, "graph.json", json.dumps(graph, indent=2), "application/json")

        md_warning = None
        try:
            id_to_title = {n["id"]: n["title"] for n in graph["nodes"] if isinstance(n, dict)}
            slug = "".join(c if c.isalnum() or c in " -_" else "" for c in new_node["title"]).strip().replace(" ", "_")
            filename = f"{new_node['id']}_{slug}.md"
            tag_line = " ".join(f"#{t}" for t in new_node["tags"])
            link_lines = [
                f"- **{lk['relation']}** [[{id_to_title.get(lk['target'], lk['target'])}]]: {lk['reason']}"
                for lk in new_links
            ]
            md = f"""# {new_node['title']}
**ID:** `{new_node['id']}` | **Created:** {new_node['created']} | **Tags:** {tag_line}

## Permanent Note
{new_node['permanent_note']}

## Literature Note
> "{new_node['quote']}"
> — *{new_node['source']}*

## Connections
{chr(10).join(link_lines) if link_lines else "_No links yet._"}
"""
            upload_text_to_drive(drive, folder_id, filename, md, "text/markdown")
        except Exception as md_err:
            md_warning = str(md_err)

        return {
            "status": "ok",
            "model_used": used_model,
            "node": {k: v for k, v in new_node.items() if k != "embedding"},
            "links": new_links,
            "md_warning": md_warning,
        }
    except Exception as e:
        traceback.print_exc()
        tb = traceback.extract_tb(e.__traceback__)
        line_info = f" (line {tb[-1].lineno})" if tb else ""
        raise HTTPException(status_code=500, detail=f"{e}{line_info}")
