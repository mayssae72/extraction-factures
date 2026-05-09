# Invoice AI Extractor

Production-oriented Streamlit app for extracting structured invoice data from:

- PDF files
- scanned PDF files with OCR fallback
- DOCX files
- images (`.png`, `.jpg`, `.jpeg`)
- raw pasted text

The app converts the document into text, sends the cleaned content to a Hugging Face model, then returns a normalized JSON payload ready for downstream processing.

## Why this version is stronger

- Multi-format ingestion instead of text-only input
- OCR fallback for scanned invoices
- Clear schema normalization for invoice data
- Better error handling and environment-based configuration
- GitHub Actions CI and unit tests
- Deploy-ready setup for Streamlit Cloud

## Extracted JSON schema

```json
{
  "document_type": "invoice",
  "invoice_number": "INV-2026-014",
  "invoice_date": "2026-04-17",
  "due_date": "2026-05-17",
  "currency": "EUR",
  "subtotal_amount": 10000.0,
  "tax_amount": 2450.0,
  "total_amount": 12450.0,
  "payment_terms": "30 days",
  "notes": null,
  "supplier": {
    "name": "Atlas Data SARL",
    "address": "Casablanca, Morocco",
    "email": "contact@atlasdata.ma",
    "phone": "+212600000000",
    "tax_id": "ICE123456789",
    "iban": "MA64..."
  },
  "customer": {
    "name": "Retail North Europe",
    "address": "Amsterdam, Netherlands",
    "email": null,
    "phone": null,
    "tax_id": null,
    "iban": null
  },
  "line_items": [
    {
      "description": "Consulting",
      "quantity": 5.0,
      "unit_price": 2000.0,
      "total": 10000.0
    }
  ]
}
```

## Local setup

```bash
git clone https://github.com/mayssae72/extraction-factures.git
cd extraction-factures
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Set your token before launching:

```bash
set HF_TOKEN=hf_your_token_here
```

Optional runtime variables are listed in [.env.example](.env.example).

## Streamlit Cloud deployment

This repo is ready for Streamlit Cloud deployment:

- `requirements.txt` installs Python dependencies
- `packages.txt` installs Tesseract for OCR
- `.streamlit/config.toml` controls theme settings

Recommended secret:

```toml
HF_TOKEN = "hf_your_token_here"
```

## Project structure

```text
extraction-factures/
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── invoice_extractor/
│   ├── config.py
│   ├── document_processing.py
│   ├── llm.py
│   ├── parsing.py
│   ├── prompts.py
│   └── schemas.py
├── tests/test_parsing.py
├── .env.example
├── app.py
├── packages.txt
├── README.md
└── requirements.txt
```

## How the pipeline works

1. Upload a document or paste raw invoice text.
2. Extract text with `pypdf`, `python-docx`, or OCR.
3. Build a strict extraction prompt.
4. Generate JSON with Hugging Face inference.
5. Normalize the payload into a predictable schema.
6. Download the result as JSON.

## GitHub portfolio value

This repository is now much better for a professional GitHub profile because it shows:

- document AI / OCR use case
- modular Python architecture
- production-minded UX with Streamlit
- test coverage for parsing logic
- CI automation with GitHub Actions
- deployment thinking, not just notebook-style code

## Good next ideas

- Add a FastAPI endpoint for API-based extraction
- Add batch processing for multiple invoices
- Export results to CSV and Excel
- Add Docker support for one-command deployment
- Create a demo dataset and benchmark page
- Add screenshots or a short GIF to the README

## License

MIT
