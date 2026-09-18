"""LLM backends for the chatbot.

Every backend implements the same tiny interface:

    backend.chat(messages, stream=False) -> str | Iterator[str]

* ``OpenAIBackend``   - any OpenAI-compatible HTTP endpoint. This covers
  OpenAI, Azure OpenAI, Groq, Together, Fireworks, vLLM, LM Studio, and Ollama
  (``http://localhost:11434/v1``). Uses only ``requests`` so no SDK is needed.
* ``HFLocalBackend``  - a real local transformers model (needs torch).
* ``EchoBackend``     - a small rule-based responder used when nothing else is
  configured, so the app always runs.

``get_backend()`` picks the best available automatically.
"""
from __future__ import annotations

import json
from typing import Iterator

from config import (BACKEND, HF_LOCAL_MODEL, MAX_TOKENS, MODEL_NAME,
                    OPENAI_API_KEY, OPENAI_BASE_URL, REQUEST_TIMEOUT,
                    SYSTEM_PROMPT, TEMPERATURE, TOP_P)


class BackendError(RuntimeError):
    pass


# --------------------------------------------------------------- OpenAI-compatible
class OpenAIBackend:
    name = "openai"

    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or MODEL_NAME
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else OPENAI_API_KEY

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages, stream=False):
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
            "stream": stream,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=REQUEST_TIMEOUT,
                                 stream=stream)
        except Exception as exc:
            raise BackendError(f"Cannot reach {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"{resp.status_code} from {url}: {resp.text[:300]}")
        if stream:
            return self._iter_stream(resp)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _iter_stream(resp) -> Iterator[str]:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"]
                piece = delta.get("content")
                if piece:
                    yield piece
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def available(self):
        try:
            import requests
            r = requests.get(f"{self.base_url}/models",
                             headers=self._headers(), timeout=5)
            return r.status_code < 500
        except Exception:
            return False


# ------------------------------------------------------------------ HF local ---
class HFLocalBackend:
    name = "hf"

    def __init__(self, model_name=None):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise BackendError("Local HF backend needs torch + transformers") from exc
        self._torch = __import__("torch")
        self.name_ = model_name or HF_LOCAL_MODEL
        self.tok = AutoTokenizer.from_pretrained(self.name_)
        self.model = AutoModelForCausalLM.from_pretrained(self.name_)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def chat(self, messages, stream=False):
        prompt = SYSTEM_PROMPT + "\n"
        for m in messages:
            prompt += f"{m['role']}: {m['content']}{self.tok.eos_token}"
        prompt += "assistant:"
        ids = self.tok.encode(prompt, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=MAX_TOKENS, pad_token_id=self.tok.eos_token_id,
                do_sample=True, top_p=TOP_P, temperature=TEMPERATURE)
        return self.tok.decode(out[:, ids.shape[-1]:][0],
                               skip_special_tokens=True).strip()

    def available(self):
        return True


# ----------------------------------------------------------------- rule-based --
class EchoBackend:
    """A small, genuinely useful rule-based assistant.

    Not a language model - it exists so the product runs, and so you can demo
    the full app (memory, RAG, API, UI) with zero downloads or API keys.
    """
    name = "echo"

    _help = ("I'm running in offline rule-based mode (no LLM configured).\n"
             "Set CHAT_BACKEND=openai with an OPENAI_API_KEY, or run Ollama "
             "locally and set CHAT_OPENAI_BASE_URL=http://localhost:11434/v1.\n"
             "Meanwhile I can: greet you, do simple math, remember facts "
             "('remember my name is ...'), and search our chat history "
             "('recall ...').")

    def chat(self, messages, stream=False):
        user = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")
        reply = self._respond(user)
        if stream:
            def gen():
                for word in reply.split():
                    yield word + " "
            return gen()
        return reply

    def _respond(self, text):
        low = text.lower().strip()
        if not low:
            return "Say something and I'll respond."
        if any(g in low for g in ("hello", "hi ", "hey")) or low in ("hi", "hey"):
            return "Hello! How can I help?"
        if "help" in low:
            return self._help
        if low.startswith("remember"):
            return "Stored that as a fact - I'll keep it in long-term memory."
        if "time" in low and "what" in low:
            import datetime
            return "Local time is " + datetime.datetime.now().strftime("%H:%M:%S")
        if "who are you" in low or "your name" in low:
            return ("I'm the local Chatbot project - a real LLM front-end with "
                    "persistent memory and RAG retrieval.")
        if "recall" in low:
            return "Use 'recall <keyword>' and I'll search our history."
        return ("(offline mode) I heard: \"" + text.strip() +
                "\". Configure an LLM backend for full answers - type 'help'.")


# ------------------------------------------------------------------- factory ---
def get_backend(preference=None):
    """Return the best available backend for the configured preference."""
    pref = (preference or BACKEND or "auto").lower()

    if pref == "echo":
        return EchoBackend()

    if pref in ("openai", "auto"):
        if OPENAI_API_KEY or pref == "openai":
            try:
                b = OpenAIBackend()
                if pref == "openai" or b.available():
                    return b
            except Exception:
                if pref == "openai":
                    raise

    if pref in ("hf", "auto"):
        try:
            return HFLocalBackend()
        except Exception:
            if pref == "hf":
                raise

    return EchoBackend()


def describe(backend) -> str:
    if isinstance(backend, OpenAIBackend):
        return f"OpenAI-compatible ({backend.model} @ {backend.base_url})"
    if isinstance(backend, HFLocalBackend):
        return f"Local HuggingFace ({backend.name_})"
    return "Offline rule-based (no LLM)"
