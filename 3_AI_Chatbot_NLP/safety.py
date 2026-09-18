"""Lightweight safety layer: PII redaction before anything is stored/sent.

This is deliberately dependency-free and conservative. It is *not* a full
moderation system - plug in a real moderation endpoint where required - but it
stops the most common accidental leaks of personal data.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_CREDIT = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact(text: str) -> str:
    """Replace common PII patterns with typed placeholders."""
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub("[SSN]", text)
    text = _CREDIT.sub("[CARD]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text


def contains_pii(text: str) -> bool:
    return bool(_EMAIL.search(text) or _PHONE.search(text)
                or _CREDIT.search(text) or _SSN.search(text))


def safe_incoming(text: str, redact_enabled=True) -> str:
    """Apply to user input before it is stored or sent to a model."""
    return redact(text) if redact_enabled else text
