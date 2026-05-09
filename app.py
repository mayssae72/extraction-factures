from __future__ import annotations

import json

import streamlit as st

from invoice_extractor.config import load_config, resolve_hf_token
from invoice_extractor.document_processing import (
    DocumentExtractionError,
    ExtractionArtifact,
    extract_text_from_upload,
)
from invoice_extractor.llm import InvoiceExtractionError, extract_invoice_data


CONFIG = load_config()
SUPPORTED_FILE_TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]
SESSION_KEYS = ["artifact", "invoice_result", "model_response"]


def _read_streamlit_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value).strip()


def _format_scalar(value: object, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _format_amount(amount: object, currency: object) -> str:
    if amount is None:
        return "-"
    try:
        rendered = f"{float(amount):,.2f}".replace(",", " ")
    except (TypeError, ValueError):
        return _format_scalar(amount)
    currency_text = _format_scalar(currency, "")
    return f"{rendered} {currency_text}".strip()


def _clear_results() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def _build_text_artifact(raw_text: str) -> ExtractionArtifact:
    return ExtractionArtifact(
        source_name="pasted_text.txt",
        source_type="Plain text",
        text=raw_text.strip(),
        warnings=[],
    )


def _render_party(title: str, party: dict[str, object]) -> None:
    st.markdown(f"#### {title}")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Nom**: {_format_scalar(party.get('name'))}")
        st.write(f"**Adresse**: {_format_scalar(party.get('address'))}")
        st.write(f"**Email**: {_format_scalar(party.get('email'))}")

    with col2:
        st.write(f"**Telephone**: {_format_scalar(party.get('phone'))}")
        st.write(f"**Tax ID**: {_format_scalar(party.get('tax_id'))}")
        st.write(f"**IBAN**: {_format_scalar(party.get('iban'))}")


def _render_result(result: dict[str, object], artifact: ExtractionArtifact, model_response: str) -> None:
    summary_cols = st.columns(4)
    summary_cols[0].metric("Type source", artifact.source_type)
    summary_cols[1].metric("Numero facture", _format_scalar(result.get("invoice_number")))
    summary_cols[2].metric("Date facture", _format_scalar(result.get("invoice_date")))
    summary_cols[3].metric(
        "Montant total",
        _format_amount(result.get("total_amount"), result.get("currency")),
    )

    meta_cols = st.columns(3)
    meta_cols[0].metric("Pages", str(artifact.page_count or 1))
    meta_cols[1].metric("OCR", "Oui" if artifact.used_ocr else "Non")
    meta_cols[2].metric("Caracteres", str(len(artifact.text)))

    if artifact.warnings:
        for warning in artifact.warnings:
            st.warning(warning)

    st.markdown("### Parties")
    party_cols = st.columns(2)
    with party_cols[0]:
        _render_party("Fournisseur", result.get("supplier", {}))
    with party_cols[1]:
        _render_party("Client", result.get("customer", {}))

    st.markdown("### Resume comptable")
    finance_cols = st.columns(4)
    finance_cols[0].metric("Sous-total", _format_amount(result.get("subtotal_amount"), result.get("currency")))
    finance_cols[1].metric("Taxes", _format_amount(result.get("tax_amount"), result.get("currency")))
    finance_cols[2].metric("Echeance", _format_scalar(result.get("due_date")))
    finance_cols[3].metric("Type doc", _format_scalar(result.get("document_type")))

    line_items = result.get("line_items", [])
    st.markdown("### Lignes facture")
    if line_items:
        st.dataframe(line_items, use_container_width=True)
    else:
        st.info("Aucune ligne detaillee n'a ete detectee.")

    notes = _format_scalar(result.get("notes"), "")
    payment_terms = _format_scalar(result.get("payment_terms"), "")
    if notes or payment_terms:
        st.markdown("### Informations complementaires")
        if payment_terms:
            st.write(f"**Conditions de paiement**: {payment_terms}")
        if notes:
            st.write(f"**Notes**: {notes}")

    json_payload = json.dumps(result, indent=2, ensure_ascii=False)

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            label="Telecharger le JSON",
            data=json_payload,
            file_name="invoice_extraction.json",
            mime="application/json",
            use_container_width=True,
        )
    with download_cols[1]:
        st.download_button(
            label="Telecharger le texte extrait",
            data=artifact.text,
            file_name="invoice_source_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("Voir le texte extrait", expanded=False):
        st.text_area("Texte extrait", value=artifact.text, height=260, disabled=True, label_visibility="collapsed")

    with st.expander("Voir le JSON complet", expanded=False):
        st.json(result)

    with st.expander("Voir la reponse brute du modele", expanded=False):
        st.code(model_response or "Aucune reponse brute disponible.", language="json")


st.set_page_config(
    page_title="Invoice AI Extractor",
    page_icon="IN",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f7fbfa 0%, #eef6f5 100%);
        border: 1px solid #d7e8e5;
        padding: 1rem;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Extraction de factures multi-format")
st.caption(
    "Version production Streamlit pour extraire des donnees facture depuis PDF, DOCX, images et texte brut."
)

with st.sidebar:
    st.header("Configuration")

    configured_token = _read_streamlit_secret("HF_TOKEN") or resolve_hf_token(None) or ""
    if configured_token:
        st.success("Un token Hugging Face est deja configure.")

    hf_token_input = st.text_input(
        "Token Hugging Face",
        type="password",
        placeholder="hf_xxxxxxxxx",
        help="Laissez vide si HF_TOKEN est defini dans l'environnement ou dans Streamlit secrets.",
    )

    hf_token = hf_token_input.strip() or configured_token

    st.markdown("### Runtime")
    st.write(f"**Modele**: `{CONFIG.hf_model}`")
    st.write(f"**OCR**: `{CONFIG.ocr_language}`")
    st.write(f"**Max input chars**: `{CONFIG.max_input_chars}`")
    st.write(f"**Max output tokens**: `{CONFIG.max_output_tokens}`")

    st.markdown("### Formats")
    st.write("PDF natif, PDF scanne, DOCX, PNG, JPG, JPEG, texte colle")

    st.markdown("### Conseils")
    st.write("Pour les images et les PDF scannes, Tesseract doit etre installe.")
    st.write("Sur Streamlit Cloud, le fichier `packages.txt` gere cette dependance.")

    st.markdown("### GitHub")
    st.write("Le repo inclut maintenant une structure plus propre, des tests et une CI.")

input_col, info_col = st.columns([1.6, 1])

with input_col:
    input_mode = st.radio(
        "Source d'entree",
        options=["Document", "Texte brut"],
        horizontal=True,
    )

    uploaded_file = None
    raw_text = ""

    if input_mode == "Document":
        uploaded_file = st.file_uploader(
            "Televersez un document",
            type=SUPPORTED_FILE_TYPES,
            help="Formats supportes: PDF, DOCX, PNG, JPG, JPEG",
        )
    else:
        raw_text = st.text_area(
            "Collez le texte de la facture",
            height=320,
            placeholder=(
                "Invoice INV-2026-014\n"
                "Date: 2026-04-17\n"
                "Supplier: Atlas Data SARL\n"
                "Customer: Retail North Europe\n"
                "Total: 12450.00 EUR"
            ),
        )

    action_cols = st.columns([2, 1])
    extract_button = action_cols[0].button("Extraire le JSON", type="primary", use_container_width=True)
    reset_button = action_cols[1].button("Reinitialiser", use_container_width=True)

with info_col:
    st.markdown("### Workflow")
    st.write("1. Charger un document ou coller le texte.")
    st.write("2. Extraire le texte avec parser natif ou OCR.")
    st.write("3. Structurer la facture en JSON via Hugging Face.")
    st.write("4. Verifier puis telecharger le resultat.")

    st.markdown("### Schema cible")
    st.code(
        '{\n'
        '  "invoice_number": "...",\n'
        '  "invoice_date": "...",\n'
        '  "total_amount": 0,\n'
        '  "currency": "...",\n'
        '  "supplier": {"name": "..."},\n'
        '  "customer": {"name": "..."},\n'
        '  "line_items": []\n'
        '}',
        language="json",
    )

if reset_button:
    _clear_results()
    st.rerun()

if extract_button:
    if not hf_token:
        st.error("Ajoutez un token Hugging Face avant de lancer l'extraction.")
    else:
        try:
            if input_mode == "Document":
                if uploaded_file is None:
                    raise DocumentExtractionError("Chargez un fichier PDF, DOCX ou image avant de lancer l'extraction.")
                with st.spinner("Lecture du document en cours..."):
                    artifact = extract_text_from_upload(
                        filename=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                        ocr_language=CONFIG.ocr_language,
                        min_direct_pdf_chars=CONFIG.min_direct_pdf_chars,
                    )
            else:
                if not raw_text.strip():
                    raise DocumentExtractionError("Collez le texte de la facture avant de lancer l'extraction.")
                artifact = _build_text_artifact(raw_text)

            if len(artifact.text) > CONFIG.max_input_chars:
                artifact.warnings.append(
                    f"Le texte envoye au modele sera tronque aux {CONFIG.max_input_chars} premiers caracteres."
                )

            with st.spinner("Structuration facture en cours..."):
                result, model_response = extract_invoice_data(
                    raw_text=artifact.text,
                    token=hf_token,
                    config=CONFIG,
                    source_name=artifact.source_name,
                )

            st.session_state["artifact"] = artifact
            st.session_state["invoice_result"] = result
            st.session_state["model_response"] = model_response
            st.success("Extraction terminee avec succes.")

        except DocumentExtractionError as exc:
            st.error(str(exc))
        except InvoiceExtractionError as exc:
            st.error(str(exc))

artifact = st.session_state.get("artifact")
result = st.session_state.get("invoice_result")
model_response = st.session_state.get("model_response", "")

st.markdown("---")
st.markdown("## Resultat")

if artifact and result:
    _render_result(result, artifact, model_response)
else:
    st.info("Aucun resultat pour le moment. Chargez un document ou collez du texte, puis lancez l'extraction.")
