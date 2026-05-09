from __future__ import annotations

import json
import re
from typing import Any

from invoice_extractor.schemas import normalize_invoice_payload


class InvoiceParsingError(ValueError):
    """Raised when the model response does not contain valid JSON."""


CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_code_fences(text: str) -> str:
    match = CODE_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def extract_json_object(text: str) -> dict[str, Any]:
    candidate_text = _normalize_quotes(_strip_code_fences(text))
    decoder = json.JSONDecoder()

    for start_index, character in enumerate(candidate_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate_text[start_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise InvoiceParsingError("No valid JSON object found in the model response.")


def parse_invoice_response(text: str) -> dict[str, Any]:
    return normalize_invoice_payload(extract_json_object(text))
