"""Chat engine: glues the LLM backend, persistent memory, RAG and safety.

``ChatEngine.ask()`` is the single entry point used by the CLI, the REST API
and the web UI, so behaviour is identical everywhere.
"""
from __future__ import annotations

from typing import Iterator, Optional

import backends
import memory
import safety
from config import MAX_HISTORY, RAG_ENABLED, RAG_TOP_K, REDACT_PII, SYSTEM_PROMPT


class ChatEngine:
    def __init__(self, conn=None, backend=None, session_id=None):
        self.conn = conn or memory.get_conn()
        self.backend = backend or backends.get_backend()
        self.session_id = session_id or memory.new_session(self.conn)
        self.retriever = memory.Retriever(self.conn) if RAG_ENABLED else None

    # ---------------------------------------------------------------- context --
    def _rag_context(self, question: str) -> str:
        if not (RAG_ENABLED and self.retriever):
            return ""
        hits = self.retriever.query(question, top_k=RAG_TOP_K)
        hits = [h for h in hits if h.get("content")]
        if not hits:
            return ""
        lines = [f"- {h['content'][:300]}" for h in hits]
        return ("Relevant things from earlier conversations you may reuse:\n"
                + "\n".join(lines))

    def _build_messages(self, question: str):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        ctx = self._rag_context(question)
        if ctx:
            msgs.append({"role": "system", "content": ctx})
        for m in memory.recent(self.conn, self.session_id, MAX_HISTORY):
            role = "assistant" if m["role"] == "bot" else m["role"]
            msgs.append({"role": role, "content": m["content"]})
        msgs.append({"role": "user", "content": question})
        return msgs

    # ------------------------------------------------------------------- ask ---
    def ask(self, question: str, remember=True, stream=False):
        """Send a question, persist both sides, return the reply (or iterator)."""
        clean = safety.safe_incoming(question, REDACT_PII)
        if remember:
            memory.add(self.conn, self.session_id, "user", clean)
        messages = self._build_messages(clean)

        if stream:
            return self._stream_reply(messages, remember)

        reply = self.backend.chat(messages)
        if remember:
            memory.add(self.conn, self.session_id, "bot", reply)
            if self.retriever:
                self.retriever.build()
        return reply

    def _stream_reply(self, messages, remember) -> Iterator[str]:
        collected = []
        try:
            for piece in self.backend.chat(messages, stream=True):
                collected.append(piece)
                yield piece
        finally:
            if remember and collected:
                memory.add(self.conn, self.session_id, "bot", "".join(collected))
                if self.retriever:
                    self.retriever.build()

    # ----------------------------------------------------------- memory cmds ---
    def recall(self, keyword: str, limit=8):
        return memory.search_text(self.conn, keyword, limit=limit)

    def remember_fact(self, key: str, value: str):
        memory.remember_fact(self.conn, self.session_id, key, value)

    def facts(self):
        return memory.recall_facts(self.conn)

    def title_from_first_message(self):
        first = memory.recent(self.conn, self.session_id, 1)
        if first:
            memory.rename_session(self.conn, self.session_id,
                                  first[0]["content"][:40])


def make_engine(backend=None, session_id: Optional[int] = None) -> ChatEngine:
    return ChatEngine(backend=backend, session_id=session_id)
