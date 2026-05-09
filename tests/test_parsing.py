from __future__ import annotations

import unittest

from invoice_extractor.parsing import extract_json_object, parse_invoice_response
from invoice_extractor.schemas import normalize_invoice_payload


class ParsingTests(unittest.TestCase):
    def test_extract_json_object_inside_code_fence(self) -> None:
        response = """The result is below.

```json
{
  "invoice_number": "INV-42",
  "total_amount": "1 240,50 EUR",
  "currency": "EUR"
}
```
"""
        payload = extract_json_object(response)
        self.assertEqual(payload["invoice_number"], "INV-42")

    def test_parse_invoice_response_normalizes_amounts(self) -> None:
        response = """
        {
          "numero_facture": "FAC-2026-009",
          "date": "2026-04-18",
          "montant_total": "12.450,99 EUR",
          "devise": "EUR",
          "fournisseur": {"nom": "Atlas Data"},
          "client": {"nom": "Retail Europe"}
        }
        """
        payload = parse_invoice_response(response)
        self.assertEqual(payload["invoice_number"], "FAC-2026-009")
        self.assertEqual(payload["invoice_date"], "2026-04-18")
        self.assertAlmostEqual(payload["total_amount"], 12450.99)
        self.assertEqual(payload["supplier"]["name"], "Atlas Data")
        self.assertEqual(payload["customer"]["name"], "Retail Europe")

    def test_normalize_invoice_payload_keeps_line_items(self) -> None:
        payload = normalize_invoice_payload(
            {
                "invoice_number": "INV-100",
                "line_items": [
                    {
                        "description": "Consulting day",
                        "quantity": "2",
                        "unit_price": "500",
                        "total": "1000",
                    }
                ],
            }
        )
        self.assertEqual(payload["invoice_number"], "INV-100")
        self.assertEqual(len(payload["line_items"]), 1)
        self.assertEqual(payload["line_items"][0]["description"], "Consulting day")
        self.assertAlmostEqual(payload["line_items"][0]["quantity"], 2.0)
        self.assertAlmostEqual(payload["line_items"][0]["unit_price"], 500.0)
        self.assertAlmostEqual(payload["line_items"][0]["total"], 1000.0)


if __name__ == "__main__":
    unittest.main()
