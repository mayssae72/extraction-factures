from __future__ import annotations

import os
from dataclasses import dataclass


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    hf_model: str
    max_input_chars: int
    max_output_tokens: int
    ocr_language: str
    min_direct_pdf_chars: int


def load_config() -> AppConfig:
    return AppConfig(
        hf_model=os.getenv("HF_TEXT_MODEL", "mistralai/Mistral-7B-Instruct-v0.1").strip(),
        max_input_chars=_read_int("MAX_INPUT_CHARS", 16000),
        max_output_tokens=_read_int("MAX_OUTPUT_TOKENS", 1400),
        ocr_language=os.getenv("OCR_LANG", "eng+fra").strip() or "eng+fra",
        min_direct_pdf_chars=_read_int("MIN_DIRECT_PDF_CHARS", 120),
    )


def resolve_hf_token(user_supplied_token: str | None) -> str | None:
    candidate = (user_supplied_token or "").strip()
    if candidate:
        return candidate

    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    return None
