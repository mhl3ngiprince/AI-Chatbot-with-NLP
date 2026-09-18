"""AI Chatbot with NLP - a real conversational assistant with memory + RAG.

Talks to any OpenAI-compatible LLM endpoint (OpenAI, Azure, Groq, vLLM,
LM Studio, Ollama) or a local HuggingFace model, keeps permanent SQLite memory,
retrieves relevant past messages (RAG), redacts PII, and ships a REST API and
web UI. Falls back to an offline rule-based responder so it always runs.

Commands
--------
    python main.py chat                 # interactive terminal chat
    python main.py ask "your question"  # one-shot
    python main.py recall <keyword>     # search memory
    python main.py history              # list sessions
    python main.py facts                # remembered facts
    python main.py api                  # REST + web UI
    python main.py selftest             # verify engine offline
"""
from __future__ import annotations

import argparse
import sys

import backends
import engine as eng
import memory
from config import API_HOST, API_PORT, REDACT_PII

HELP = """slash commands:
  /recall <kw>          search all past messages
  /remember k = v       store a fact in long-term memory
  /facts                list stored facts
  /history              list sessions
  /backend              show which LLM backend is active
  /help /  /quit
"""


def _banner(chat: eng.ChatEngine):
    print("=" * 62)
    print("AI Chatbot  ·  backend: " + backends.describe(chat.backend))
    print("session #%d  ·  PII redaction: %s  ·  RAG: %s"
          % (chat.session_id, "on" if REDACT_PII else "off",
             "on" if chat.retriever else "off"))
    print("=" * 62)
    print(HELP)


def cmd_chat(args):
    chat = eng.make_engine()
    _banner(chat)
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.startswith("/"):
            if _slash(user, chat):
                continue
            break
        try:
            print("Bot: ", end="", flush=True)
            for piece in chat.ask(user, stream=True):
                print(piece, end="", flush=True)
            print()
        except backends.BackendError as exc:
            print(f"\n[backend error] {exc}")
    print("\nGoodbye - everything is saved in the database.")


def _slash(user, chat):
    """Handle a slash command. Returns True to continue, False to quit."""
    cmd, _, rest = user[1:].partition(" ")
    cmd = cmd.lower()
    if cmd in ("quit", "exit", "q"):
        return False
    if cmd == "help":
        print(HELP)
    elif cmd == "recall":
        rows = chat.recall(rest.strip())
        for r in rows:
            print(f"  [{r['role']}] {r['content'][:120]}")
        if not rows:
            print("  (nothing found)")
    elif cmd == "remember":
        key, _, val = rest.partition("=")
        chat.remember_fact(key.strip(), val.strip())
        print("  stored.")
    elif cmd == "facts":
        for f in chat.facts():
            print(f"  {f['key']} = {f['value']}")
    elif cmd == "history":
        _print_sessions(chat)
    elif cmd == "backend":
        print("  " + backends.describe(chat.backend))
    else:
        print("  unknown command - /help")
    return True


def cmd_ask(args):
    chat = eng.make_engine()
    try:
        print(chat.ask(" ".join(args.question)))
    except backends.BackendError as exc:
        sys.exit(f"[backend error] {exc}")


def cmd_recall(args):
    chat = eng.make_engine()
    for r in chat.recall(" ".join(args.keyword)):
        print(f"[{r['role']}] {r['content']}")


def cmd_history(args):
    chat = eng.make_engine()
    _print_sessions(chat)


def _print_sessions(chat):
    for s in memory.sessions(chat.conn):
        title = s["title"] or "(untitled)"
        print(f"  #{s['id']:<4} {title[:40]:<42} {s['n']} messages")
    print("  totals:", memory.stats(chat.conn))


def cmd_facts(args):
    chat = eng.make_engine()
    for f in chat.facts():
        print(f"  {f['key']} = {f['value']}")


def cmd_api(args):
    try:
        import uvicorn
    except ImportError:
        sys.exit("Install API extras: pip install fastapi 'uvicorn[standard]'")
    print(f"Chat API + web UI on http://{API_HOST}:{API_PORT}")
    uvicorn.run("chat_api:app", host=API_HOST, port=API_PORT, reload=False)


def cmd_selftest(args):
    print("== Self-test (offline) ==")
    import tempfile
    conn = memory.get_conn(tempfile.mktemp(suffix=".db"))
    chat = eng.ChatEngine(conn=conn, backend=backends.EchoBackend())
    print("  backend     :", backends.describe(chat.backend))
    r1 = chat.ask("Hello there")
    print("  ask()       :", r1[:60])
    chat.remember_fact("name", "Ada")
    print("  facts       :", chat.facts())
    r2 = chat.ask("what did I say earlier")
    print("  rag ask()   :", r2[:60])
    hits = chat.recall("Hello")
    print("  recall hits :", len(hits))
    print("  stats       :", memory.stats(conn))

    # safety
    import safety
    red = safety.redact("email me at a@b.com or 082 123 4567")
    print("  redaction   :", red)
    ok = bool(r1) and hits and "[EMAIL]" in red
    print("  SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("chat").set_defaults(func=cmd_chat)
    a = sub.add_parser("ask"); a.add_argument("question", nargs="+"); a.set_defaults(func=cmd_ask)
    r = sub.add_parser("recall"); r.add_argument("keyword", nargs="+"); r.set_defaults(func=cmd_recall)
    sub.add_parser("history").set_defaults(func=cmd_history)
    sub.add_parser("facts").set_defaults(func=cmd_facts)
    sub.add_parser("api").set_defaults(func=cmd_api)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
