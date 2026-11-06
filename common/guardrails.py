"""Guardrails shared by the sub-agent and the orchestrator (R4a, R4b).

Everything that comes back from a tool is untrusted data. Two transforms:
  * redact_pii        - emails / phone numbers -> placeholders
  * neutralise_injection - instruction-like text -> placeholder (withheld entirely)

Both are idempotent, so applying them at several choke points is safe.
"""
import re
from typing import Any

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w])")

# Structured fields that are never PII or free text: left untouched so that
# dates ("2026-11-03T10:00:00") are not mistaken for phone numbers.
STRUCTURAL_KEYS = {"id", "status", "total", "placed_at", "merchant", "ok", "code"}
# Fields that are PII by definition: replaced wholesale.
PII_FIELDS = {"customer_email": "[REDACTED_EMAIL]", "customer_phone": "[REDACTED_PHONE]"}

INJECTION_PLACEHOLDER = "[removed: instruction-like text in customer-provided content]"
INJECTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?|polic(?:y|ies))\b",
    r"\b(system|developer|assistant|admin)\s*(override|prompt|mode|message)\b",
    r"^\s*(assistant|system|developer)\s*:",
    r"</?\s*(instructions?|system|assistant|prompt)\s*>",
    r"\byou are now\b",
    r"\b(refund|cancel)\b.{0,20}\b(this|every|all|the)\b.{0,15}\borders?\b.{0,30}\b(immediately|now|without)\b",
    r"\b(refund|cancel) (this|every|all)( \w+)? orders?\b",
    r"\bwithout (asking|approval|confirmation|the staff)\b",
    r"\breveal\b.{0,40}\b(emails?|phones?|passwords?|tokens?|secrets?|prompt)\b",
)]


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)

    def _phone(m: re.Match) -> str:
        return "[REDACTED_PHONE]" if sum(c.isdigit() for c in m.group()) >= 9 else m.group()

    return PHONE_RE.sub(_phone, text)


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def neutralise_injection(text: str) -> str:
    return INJECTION_PLACEHOLDER if looks_like_injection(text) else text


def sanitize_text(text: str) -> str:
    """Full treatment for any untrusted string."""
    return redact_pii(neutralise_injection(text))


def sanitize_data(value: Any, key: str | None = None) -> Any:
    """Recursively sanitise decoded JSON (tool output)."""
    if isinstance(value, dict):
        return {k: sanitize_data(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_data(v, key) for v in value]
    if isinstance(value, str):
        if key in PII_FIELDS:
            return PII_FIELDS[key]
        if key in STRUCTURAL_KEYS:
            return value
        return sanitize_text(value)
    return value
