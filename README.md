# 3 · AI Chatbot with NLP - Real LLM, Persistent Memory, RAG, Web UI

A production-shaped conversational assistant. It talks to a **real LLM**
(any OpenAI-compatible endpoint, or a local HuggingFace model), keeps
**permanent SQLite memory**, retrieves relevant past conversation via **RAG**,
**redacts PII** before anything is stored, and ships a **REST API + web UI**.

It also runs with **zero configuration** using an offline rule-based responder,
so the whole product (memory, RAG, API, UI, safety) is demonstrable instantly.

## What's real here

| Concern | What this does |
|--------|-----------------|
| LLM | OpenAI-compatible HTTP backend -> OpenAI, Azure, Groq, Together, vLLM, LM Studio, **Ollama**; or local `transformers` |
| Memory | Every session + message persisted in SQLite; survives restarts |
| RAG | Embeds & ranks past messages - `sentence-transformers` if present, else TF-IDF cosine, else substring |
| Context | System prompt + retrieved memory + last N turns -> model |
| Safety | Email/phone/card/SSN redaction *before* storage and before sending |
| Streaming | Token streaming in the terminal |
| Interface | Interactive CLI **and** browser chat UI **and** REST API |
| Robustness | Falls back gracefully; never hard-fails on a missing model |
| Tests | `pytest` suite for memory, RAG, engine, safety, backends |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: set an LLM backend

python main.py selftest       # verify with no model/keys
python main.py chat           # interactive terminal chat
python main.py api            # web UI + API at http://127.0.0.1:8003
```

### Use a real LLM

**OpenAI / Azure / Groq / Together (any OpenAI-compatible API)**
```bash
# .env
CHAT_BACKEND=openai
CHAT_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**Local Ollama (no API key, fully private)**
```bash
ollama pull llama3.1
# .env
CHAT_BACKEND=openai
CHAT_MODEL=llama3.1
CHAT_OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

**Local HuggingFace**
```bash
pip install torch transformers
# .env
CHAT_BACKEND=hf
CHAT_HF_MODEL=microsoft/DialoGPT-medium
```

## Commands
```
chat                 interactive chat
ask "<question>"     one-shot
recall <keyword>     search memory
history              list sessions
facts                list remembered facts
api                  run REST API + web UI
selftest             verify offline
```
In chat: `/recall`, `/remember k = v`, `/facts`, `/history`, `/backend`, `/help`.

## REST API
| Method | Path | Purpose |
|-------|------|---------|
| GET  | `/health` | backend + DB status |
| POST | `/ask` | ask a question (persists + RAG) |
| GET  | `/sessions` · `/history/{id}` | conversation history |
| GET  | `/recall?q=` | search memory |
| POST | `/facts` · GET `/facts` | long-term facts |

## Tables
`sessions` · `messages` · `facts`

## Notes & extensions
* Point `CHAT_EMBED_MODEL` + install `sentence-transformers` for stronger RAG.
* Add real moderation by calling a moderation endpoint in `safety.py`.
* For multi-user deployments, add auth and per-user namespaces (the schema
  already keys everything by `session_id`).

> Privacy: with a cloud backend your text leaves the machine. Use Ollama/HF for
> fully local operation.

## Web dashboard

Every project ships a server-rendered **web dashboard** (a real HTML page).

* **No emoji** ? all icons are inline **SVG** (defined in `../_shared/dashboard_kit.py`).
* **Real data only** ? every card is labelled with its data source, and the
  page reads the same SQLite tables the pipeline writes.
* **No CDN / no JavaScript required** ? charts are plain inline SVG.

Open it by running the project's API and visiting `/dashboard`:

    python <api-entrypoint>            # start the service
    # then open http://127.0.0.1:<port>/dashboard
