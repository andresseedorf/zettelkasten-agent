import os
import io
import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

app = FastAPI()
client = genai.Client()  # Reads GEMINI_API_KEY automatically

DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

creds_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
creds = service_account.Credentials.from_service_account_info(
    creds_info, scopes=["https://www.googleapis.com/auth/drive"]
)
drive_service = build("drive", "v3", credentials=creds)


class EdgeProposal(BaseModel):
    target_id: str
    relation: Literal["supports", "contradicts", "extends", "analogous_to"]
    reason: str = Field(description="One concise sentence explaining the conceptual link.")


class ZettelSynthesis(BaseModel):
    title: str = Field(description="Short, declarative concept title (3-7 words).")
    permanent_note: str = Field(description="Atomic idea rewritten in clear, standalone prose.")
    tags: List[str] = Field(description="2-4 lowercase topical tags without #.")
    connections: List[EdgeProposal] = Field(
        description="Meaningful conceptual links to candidate notes. Leave empty if unrelated."
    )


class NoteInput(BaseModel):
    quote: str = ""
    thought: str
    source: str = "Unattributed"


def verify_auth(x_passcode: str = Header(default="")):
    if APP_PASSWORD and x_passcode != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid passcode")


# --- Google Drive Persistence ---
def find_drive_file(filename: str):
    q = f"name = '{filename}' and '{DRIVE_FOLDER_ID}' in parents and trashed = false"
    res = drive_service.files().list(q=q, fields="files(id, name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def load_graph() -> dict:
    file_id = find_drive_file("graph.json")
    if not file_id:
        return {"nodes": [], "links": []}
    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return json.loads(buf.getvalue().decode("utf-8"))


def upload_text_to_drive(filename: str, content: str, mime_type: str):
    media = MediaIoBaseUpload(io.BytesIO(content.encode("utf-8")), mimetype=mime_type, resumable=False)
    existing_id = find_drive_file(filename)
    if existing_id:
        drive_service.files().update(fileId=existing_id, media_body=media).execute()
    else:
        meta = {"name": filename, "parents": [DRIVE_FOLDER_ID]}
        drive_service.files().create(body=meta, media_body=media, fields="id").execute()


def save_graph(data: dict):
    upload_text_to_drive("graph.json", json.dumps(data, indent=2), "application/json")


def write_markdown_to_drive(node: dict, links: List[dict], id_to_title: dict):
    slug = "".join(c if c.isalnum() or c in " -_" else "" for c in node["title"]).strip().replace(" ", "_")
    filename = f"{node['id']}_{slug}.md"
    tag_line = " ".join(f"#{t}" for t in node["tags"])

    link_lines = []
    for lk in links:
        target_title = id_to_title.get(lk["target"], lk["target"])
        link_lines.append(f"- **{lk['relation']}** [[{target_title}]]: {lk['reason']}")

    md = f"""# {node['title']}
**ID:** `{node['id']}` | **Created:** {node['created']} | **Tags:** {tag_line}

## Permanent Note
{node['permanent_note']}

## Literature Note
> "{node['quote']}"
> — *{node['source']}*

## Connections
{chr(10).join(link_lines) if link_lines else "_No links yet._"}
"""
    upload_text_to_drive(filename, md, "text/markdown")


# --- Semantic Search & Synthesis ---
def get_embedding(text: str) -> List[float]:
    res = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return res.embeddings[0].values


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / denom if denom else 0.0


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return Path("index.html").read_text(encoding="utf-8")


@app.get("/api/graph")
def read_graph(x_passcode: str = Header(default="")):
    verify_auth(x_passcode)
    graph = load_graph()
    clean_nodes = [{k: v for k, v in n.items() if k != "embedding"} for n in graph["nodes"]]
    return {"nodes": clean_nodes, "links": graph["links"]}


@app.post("/api/notes")
def add_note(payload: NoteInput, x_passcode: str = Header(default="")):
    verify_auth(x_passcode)
    graph = load_graph()
    raw_text = f"Quote: {payload.quote}\nSource: {payload.source}\nThought: {payload.thought}"
    new_emb = get_embedding(raw_text)

    scored = [
        (cosine_sim(new_emb, n["embedding"]), n)
        for n in graph["nodes"]
        if "embedding" in n
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [
        {"id": n["id"], "title": n["title"], "permanent_note": n["permanent_note"]}
        for _, n in scored[:6]
    ]

    prompt = f"""You are a Zettelkasten research assistant.
Distill the user's input into ONE atomic Permanent Note, and evaluate whether it meaningfully connects to any of the candidate existing notes.

NEW INPUT:
{raw_text}

CANDIDATE EXISTING NOTES:
{json.dumps(candidates, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ZettelSynthesis,
            temperature=0.2,
        ),
    )
    synthesis = ZettelSynthesis.model_validate_json(response.text)

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

    valid_ids = {n["id"] for n in graph["nodes"]}
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

    save_graph(graph)
    id_to_title = {n["id"]: n["title"] for n in graph["nodes"]}
    write_markdown_to_drive(new_node, new_links, id_to_title)

    return {"status": "ok", "node": {k: v for k, v in new_node.items() if k != "embedding"}, "links": new_links}
