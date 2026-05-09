from __future__ import annotations

from invoice_extractor.config import AppConfig
from invoice_extractor.parsing import InvoiceParsingError, parse_invoice_response
from invoice_extractor.prompts import build_invoice_prompt


class InvoiceExtractionError(RuntimeError):
    """Raised when the structured extraction pipeline fails."""


def extract_invoice_data(
    raw_text: str,
    token: str,
    config: AppConfig,
    source_name: str,
) -> tuple[dict[str, object], str]:
    cleaned_text = raw_text.strip()
    if not cleaned_text:
        raise InvoiceExtractionError("No text is available for structured extraction.")

    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise InvoiceExtractionError("The package 'huggingface-hub' is not installed.") from exc

    prompt = build_invoice_prompt(
        source_name=source_name,
        raw_text=cleaned_text,
        max_input_chars=config.max_input_chars,
    )

    client = InferenceClient(model=config.hf_model, token=token)

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=config.max_output_tokens,
            temperature=0.1,
            return_full_text=False,
        )
    except TypeError:
        response = client.text_generation(
            prompt,
            max_new_tokens=config.max_output_tokens,
            temperature=0.1,
        )
    except Exception as exc:
        raise InvoiceExtractionError(f"Hugging Face request failed: {exc}") from exc

    if not response or not response.strip():
        raise InvoiceExtractionError("The model returned an empty response.")

    try:
        return parse_invoice_response(response), response
    except InvoiceParsingError as exc:
        raise InvoiceExtractionError(
            "The model answered, but the JSON could not be parsed. "
            "Try a cleaner source document or a stronger model."
        ) from exc
