"""Server-rendered dashboard for the chatbot.

No emoji (SVG icons only). Shows real sessions, messages, RAG corpus size and
long-term facts from the local SQLite memory.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))

import dashboard_kit as kit   # noqa: E402
import memory                 # noqa: E402


def render(conn=None) -> str:
    conn = conn or memory.get_conn()

    stats = memory.stats(conn)
    sessions = memory.sessions(conn, limit=15)
    facts = memory.recall_facts(conn, limit=20)
    # recent exchange pairs
    recent = conn.execute(
        "SELECT session_id, role, content, ts FROM messages"
        " ORDER BY id DESC LIMIT 20").fetchall()

    # how much of the memory is ingested into RAG
    corpus = conn.execute(
        "SELECT COUNT(*) n FROM messages WHERE role IN ('user','bot')"
    ).fetchone()["n"]

    session_rows = [(s["id"], (s["title"] or "(untitled)")[:40],
                     s["n"], _ts(s["started"])) for s in sessions]

    cards = [
        kit.card("Memory", "database",
                 kit.stat("Sessions", stats["sessions"], "", "chat")
                 + kit.stat("Messages", stats["messages"], "", "list")
                 + kit.stat("Stored facts", stats["facts"], "", "info"),
                 note="Source: sessions / messages / facts tables (chat.db)"),
        kit.card("Retrieval (RAG)", "brain",
                 kit.stat("Indexed messages", corpus, "", "brain")
                 + '<p class="muted">Semantic retrieval uses a real embedding '
                   'model (sentence-transformers) when available, otherwise '
                   'TF-IDF cosine.</p>',
                 note="Source: messages table used as the RAG corpus"),
        kit.card("Sessions", "chat",
                 kit.table(["id", "title", "messages", "started"],
                           session_rows),
                 span=2,
                 note="Source: sessions table"),
        kit.card("Long-term facts", "info",
                 kit.table(["key", "value"],
                           [(f["key"], f["value"]) for f in facts]),
                 note="Source: facts table"),
        kit.card("Recent conversation", "list",
                 kit.table(["session", "role", "message"],
                           [(m["session_id"], m["role"],
                             (m["content"] or "")[:80]) for m in recent]),
                 span=2,
                 note="Source: messages table (most recent first)"),
    ]
    return kit.page("AI Chatbot", "Conversational assistant with memory and RAG",
                    "".join(cards),
                    footer="Data source: local chat.db (no cloud required)")


def _ts(epoch):
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(epoch)).strftime(
            "%Y-%m-%d %H:%M")
    except Exception:
        return "-"
