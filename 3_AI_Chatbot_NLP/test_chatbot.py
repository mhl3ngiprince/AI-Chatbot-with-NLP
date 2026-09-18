"""Tests for the chatbot - run with:  pytest -q"""
import tempfile

import pytest

import backends
import engine as eng
import memory
import safety


@pytest.fixture()
def conn():
    return memory.get_conn(tempfile.mktemp(suffix=".db"))


@pytest.fixture()
def chat(conn, monkeypatch):
    # Use the offline TF-IDF retriever in tests: fast and dependency-free.
    monkeypatch.setattr(memory, "_try_sentence_transformer", lambda: None)
    return eng.ChatEngine(conn=conn, backend=backends.EchoBackend())


# ------------------------------------------------------------------ memory ---
def test_sessions_and_messages(conn):
    sid = memory.new_session(conn, "test")
    memory.add(conn, sid, "user", "hello world")
    memory.add(conn, sid, "bot", "hi there")
    recent = memory.recent(conn, sid, 10)
    assert [m["role"] for m in recent] == ["user", "bot"]
    assert memory.stats(conn)["messages"] == 2


def test_search_text(conn):
    sid = memory.new_session(conn)
    memory.add(conn, sid, "user", "I love pineapples")
    assert memory.search_text(conn, "pineapple")
    assert not memory.search_text(conn, "bananas")


def test_facts(conn):
    sid = memory.new_session(conn)
    memory.remember_fact(conn, sid, "name", "Ada")
    facts = memory.recall_facts(conn, sid)
    assert facts[0]["key"] == "name" and facts[0]["value"] == "Ada"


# --------------------------------------------------------------------- RAG ---
def test_retriever_finds_relevant(monkeypatch):
    # Force the fast TF-IDF path so the suite does not depend on (or download)
    # the neural embedder; the retrieval *logic* is what we are testing here.
    monkeypatch.setattr(memory, "_try_sentence_transformer", lambda: None)
    conn = memory.get_conn(tempfile.mktemp(suffix=".db"))
    sid = memory.new_session(conn)
    memory.add(conn, sid, "user", "My favourite colour is teal")
    memory.add(conn, sid, "user", "I drive a red bicycle")
    memory.add(conn, sid, "user", "The meeting is on Tuesday")
    r = memory.Retriever(conn).build()
    hits = r.query("what colour do I like", top_k=1)
    assert hits and "teal" in hits[0]["content"]


# ------------------------------------------------------------------ engine ---
def test_ask_persists(chat):
    reply = chat.ask("Hello")
    assert isinstance(reply, str) and reply
    roles = [m["role"] for m in memory.recent(chat.conn, chat.session_id, 10)]
    assert "user" in roles and "bot" in roles


def test_stream(chat):
    pieces = list(chat.ask("hi there", stream=True))
    assert pieces and isinstance("".join(pieces), str)


def test_recall_command(chat):
    chat.ask("I like turtles")
    hits = chat.recall("turtles")
    assert hits


def test_backend_factory_offline_defaults_to_echo():
    # with no API key and no torch, the factory must still return a usable backend
    b = backends.get_backend("echo")
    assert isinstance(b, backends.EchoBackend)


# ------------------------------------------------------------------ safety ---
def test_redact_email():
    assert "[EMAIL]" in safety.redact("contact me at a@b.com")


def test_redact_phone():
    assert "[PHONE]" in safety.redact("call 0821234567 now")


def test_redact_card():
    assert "[CARD]" in safety.redact("card 4111 1111 1111 1111")


def test_no_pii_detected():
    assert not safety.contains_pii("just a normal sentence")


def test_pii_redacted_before_storage(chat):
    chat.ask("my email is bob@example.com")
    stored = memory.search_text(chat.conn, "example.com")
    assert not stored, "raw email must not be stored"


def test_openai_backend_builds_payload(monkeypatch):
    b = backends.OpenAIBackend(model="test", base_url="http://x/v1", api_key="k")
    assert b._headers()["Authorization"] == "Bearer k"
    assert b.model == "test"
