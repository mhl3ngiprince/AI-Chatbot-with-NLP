"""FastAPI service + web UI for the chatbot.

Run:  python main.py api      (UI + docs at http://127.0.0.1:8003)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import backends
import engine as eng
import memory

app = FastAPI(title="Chatbot API", version="2.0.0")

_engine: Optional[eng.ChatEngine] = None


def engine() -> eng.ChatEngine:
    global _engine
    if _engine is None:
        _engine = eng.make_engine()
    return _engine


class AskIn(BaseModel):
    message: str
    session_id: Optional[int] = None
    remember: bool = True


class FactIn(BaseModel):
    key: str
    value: str


@app.get("/health")
def health():
    chat = engine()
    return {"status": "ok", "backend": backends.describe(chat.backend),
            "session_id": chat.session_id, "db": memory.stats(chat.conn)}


@app.post("/ask")
def ask(body: AskIn):
    chat = engine()
    if body.session_id and body.session_id != chat.session_id:
        chat.session_id = body.session_id
    try:
        reply = chat.ask(body.message, remember=body.remember)
    except backends.BackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"reply": reply, "session_id": chat.session_id}


@app.get("/sessions")
def sessions():
    return {"sessions": memory.sessions(engine().conn)}


@app.get("/history/{session_id}")
def history(session_id: int, limit: int = 50):
    rows = memory.recent(engine().conn, session_id, limit)
    return {"session_id": session_id, "messages": rows}


@app.get("/recall")
def recall(q: str, limit: int = 8):
    return {"matches": engine().recall(q, limit)}


@app.post("/facts")
def add_fact(body: FactIn):
    engine().remember_fact(body.key, body.value)
    return {"ok": True}


@app.get("/facts")
def get_facts():
    return {"facts": engine().facts()}


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI_HTML
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Server-rendered analytics dashboard (SVG icons, real data only)."""
    import dashboard as dash
    return dash.render(engine().conn)


_UI_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AI Chatbot</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#0f1115;color:#e8e8e8}
 header{padding:14px 18px;background:#171a21;border-bottom:1px solid #242833;font-weight:600}
 #log{padding:18px;height:calc(100vh - 150px);overflow:auto}
 .m{margin:8px 0;padding:10px 14px;border-radius:12px;max-width:70%;white-space:pre-wrap}
 .you{background:#1f6feb;margin-left:auto}
 .bot{background:#242833}
 form{display:flex;gap:8px;padding:14px;background:#171a21;border-top:1px solid #242833}
 input{flex:1;padding:12px;border-radius:10px;border:1px solid #2c3140;background:#0f1115;color:#e8e8e8}
 button{padding:12px 20px;border:0;border-radius:10px;background:#1f6feb;color:#fff;font-weight:600;cursor:pointer}
 small{color:#8b93a7}
</style></head>
<body>
<header>AI Chatbot <small id="meta">connecting...</small></header>
<div id="log"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Ask anything... (try /recall <kw>)"/><button>Send</button></form>
<script>
const log=document.getElementById('log');
function add(cls,text){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
async function meta(){try{const r=await fetch('/health');const j=await r.json();document.getElementById('meta').textContent=j.backend;}catch(e){}}
meta();
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const q=document.getElementById('q').value.trim();if(!q)return;
 add('you',q);document.getElementById('q').value='';
 const bubble=add('bot','...');
 try{
   const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});
   const j=await r.json();
   bubble.textContent = r.ok ? j.reply : ('Error: '+(j.detail||r.status));
 }catch(err){bubble.textContent='Network error: '+err;}
});
</script></body></html>"""
