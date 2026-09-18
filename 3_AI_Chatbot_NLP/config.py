"""Central configuration for the chatbot service."""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name, default):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------ backend ---
# "openai"  -> any OpenAI-compatible HTTP endpoint (OpenAI, Azure, Groq,
#              Together, vLLM, LM Studio, Ollama's OpenAI shim, ...)
# "hf"      -> local HuggingFace transformers model (needs torch + transformers)
# "echo"    -> dependency-free rule-based responder (always works)
BACKEND = os.getenv("CHAT_BACKEND", "auto").lower()
MODEL_NAME = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# OpenAI-compatible endpoint. For local Ollama: http://localhost:11434/v1
OPENAI_BASE_URL = os.getenv("CHAT_OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("CHAT_API_KEY", ""))

# Sampling
TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "400"))
TOP_P = float(os.getenv("CHAT_TOP_P", "0.95"))
REQUEST_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", "60"))

# Local HF options
HF_LOCAL_MODEL = os.getenv("CHAT_HF_MODEL", "microsoft/DialoGPT-medium")

# ---------------------------------------------------------------- memory/RAG --
DB_PATH = os.getenv("CHAT_DB_PATH", "chat.db")
MAX_HISTORY = int(os.getenv("CHAT_MAX_HISTORY", "10"))   # turns sent to the model
RAG_ENABLED = _bool("CHAT_RAG", "true")
RAG_TOP_K = int(os.getenv("CHAT_RAG_TOP_K", "4"))
EMBED_MODEL = os.getenv("CHAT_EMBED_MODEL", "all-MiniLM-L6-v2")

# ------------------------------------------------------------------- safety ---
REDACT_PII = _bool("CHAT_REDACT_PII", "true")
SYSTEM_PROMPT = os.getenv(
    "CHAT_SYSTEM_PROMPT",
    "You are a helpful, concise, honest assistant running on the user's own "
    "machine. If you are unsure, say so. Never invent facts or citations.")

# --------------------------------------------------------------------- API ----
API_HOST = os.getenv("CHAT_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("CHAT_API_PORT", "8003"))
