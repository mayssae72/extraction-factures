from __future__ import annotations

import json
from textwrap import dedent

from invoice_extractor.schemas import INVOICE_TEMPLATE


def build_invoice_prompt(source_name: str, raw_text: str, max_input_chars: int) -> str:
    excerpt = raw_text[:max_input_chars].strip()
    schema = json.dumps(INVOICE_TEMPLATE, indent=2)

    return dedent(
        f"""
        You are a production-grade invoice extraction system.
        Extract structured data from the source content and return valid JSON only.

        Rules:
        - Follow the schema exactly.
        - Use null for unknown scalar values.
        - Use [] for unknown lists.
        - Do not invent values that are not present in the source.
        - Keep dates and identifiers close to the original formatting.
        - If the document looks like an invoice, keep document_type as "invoice".

        JSON schema:
        {schema}

        Source name: {source_name}
        Source content:
        {excerpt}
        """
    ).strip()
