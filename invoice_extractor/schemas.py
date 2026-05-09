from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


INVOICE_TEMPLATE = {
    "document_type": "invoice",
    "invoice_number": None,
    "invoice_date": None,
    "due_date": None,
    "currency": None,
    "subtotal_amount": None,
    "tax_amount": None,
    "total_amount": None,
    "payment_terms": None,
    "notes": None,
    "supplier": {
        "name": None,
        "address": None,
        "email": None,
        "phone": None,
        "tax_id": None,
        "iban": None,
    },
    "customer": {
        "name": None,
        "address": None,
        "email": None,
        "phone": None,
        "tax_id": None,
        "iban": None,
    },
    "line_items": [],
}


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_amount(value: Any) -> float | None:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    compact = text.replace("\u00a0", "").replace(" ", "")
    compact = re.sub(r"[^0-9,.\-]", "", compact)
    if not compact:
        return None

    if compact.count(",") and compact.count("."):
        if compact.rfind(",") > compact.rfind("."):
            compact = compact.replace(".", "").replace(",", ".")
        else:
            compact = compact.replace(",", "")
    elif compact.count(",") == 1 and compact.count(".") == 0:
        compact = compact.replace(",", ".")
    elif compact.count(".") > 1:
        head, tail = compact.rsplit(".", 1)
        compact = head.replace(".", "") + "." + tail

    try:
        return float(compact)
    except ValueError:
        return None


def _normalize_party(payload: Any) -> dict[str, str | None]:
    if not isinstance(payload, dict):
        payload = {}

    return {
        "name": _normalize_string(payload.get("name") or payload.get("nom")),
        "address": _normalize_string(payload.get("address") or payload.get("adresse")),
        "email": _normalize_string(payload.get("email")),
        "phone": _normalize_string(payload.get("phone") or payload.get("telephone")),
        "tax_id": _normalize_string(payload.get("tax_id") or payload.get("ice") or payload.get("vat_number")),
        "iban": _normalize_string(payload.get("iban")),
    }


def _normalize_line_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    normalized = {
        "description": _normalize_string(item.get("description") or item.get("label") or item.get("designation")),
        "quantity": _normalize_amount(item.get("quantity") or item.get("qty")),
        "unit_price": _normalize_amount(item.get("unit_price") or item.get("prix_unitaire")),
        "total": _normalize_amount(item.get("total") or item.get("amount") or item.get("montant")),
    }

    if any(value is not None for value in normalized.values()):
        return normalized
    return None


def normalize_invoice_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    normalized = deepcopy(INVOICE_TEMPLATE)
    normalized["document_type"] = _normalize_string(payload.get("document_type")) or "invoice"
    normalized["invoice_number"] = _normalize_string(
        payload.get("invoice_number") or payload.get("numero_facture") or payload.get("invoice_id")
    )
    normalized["invoice_date"] = _normalize_string(payload.get("invoice_date") or payload.get("date"))
    normalized["due_date"] = _normalize_string(payload.get("due_date") or payload.get("date_echeance"))
    normalized["currency"] = _normalize_string(payload.get("currency") or payload.get("devise"))
    normalized["subtotal_amount"] = _normalize_amount(payload.get("subtotal_amount") or payload.get("subtotal"))
    normalized["tax_amount"] = _normalize_amount(payload.get("tax_amount") or payload.get("taxes"))
    normalized["total_amount"] = _normalize_amount(payload.get("total_amount") or payload.get("montant_total"))
    normalized["payment_terms"] = _normalize_string(payload.get("payment_terms") or payload.get("conditions_paiement"))
    normalized["notes"] = _normalize_string(payload.get("notes") or payload.get("commentaires"))
    normalized["supplier"] = _normalize_party(payload.get("supplier") or payload.get("fournisseur"))
    normalized["customer"] = _normalize_party(payload.get("customer") or payload.get("client"))

    line_items = payload.get("line_items") or payload.get("items") or payload.get("lignes") or []
    if isinstance(line_items, list):
        normalized["line_items"] = [
            normalized_item
            for item in line_items
            if (normalized_item := _normalize_line_item(item)) is not None
        ]

    return normalized
