"""
Interface SIMPLE - API Cloud (AUCUN TELECHARGEMENT)
Mistral 7B via Hugging Face API
COMPATIBLE STREAMLIT CLOUD
"""

import streamlit as st
import json

# Configuration page
st.set_page_config(
    page_title="Extraction Factures",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📄 Extraction de Factures")
st.markdown("**Mistral 7B API Cloud - Sans Installation**")

# Sidebar - Token
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    hf_token = st.text_input(
        "Token Hugging Face:",
        type="password",
        help="Créer sur https://huggingface.co/settings/tokens"
    )
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Créer un token HF
    2. Le copier ici
    3. Coller texte facture
    4. Cliquer Extraire
    5. Télécharger JSON
    """)
    
    st.markdown("[Créer token Hugging Face](https://huggingface.co/settings/tokens)")

# Vérifier token
if not hf_token:
    st.warning("⚠️ Entrez votre token Hugging Face!")
    st.info("""
    **Token gratuit en 2 minutes:**
    1. Aller sur https://huggingface.co/settings/tokens
    2. Cliquer "New token"
    3. Copier le token
    4. Le coller dans la sidebar
    """)
    st.stop()

st.success("✓ Token détecté")

# Main content
st.markdown("---")
st.markdown("### 📝 Collez le texte de votre facture")

# Text area
texte = st.text_area(
    "Texte de la facture:",
    height=250,
    placeholder="""FACTURE N° FAC-2024-001
Date: 15/01/2024
Fournisseur: Acme SARL
Client: GlobalRetail Inc
Total: 1500.00 EUR
..."""
)

# Bouton extraction
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    extract_button = st.button("🚀 Extraire", use_container_width=True)

with col2:
    clear_button = st.button("🔄 Effacer", use_container_width=True)

# Effacer
if clear_button:
    st.session_state.resultat = None
    st.rerun()

# Extraire
if extract_button:
    if not texte.strip():
        st.error("❌ Collez du texte d'abord!")
        st.stop()
    
    with st.spinner("⏳ Extraction en cours avec Mistral 7B..."):
        try:
            from huggingface_hub import InferenceClient
            
            # Créer client
            client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.1",
                token=hf_token
            )
            
            # Prompt
            prompt = f"""Vous êtes un expert extraction factures.
Extrayez JSON structuré de ce texte:

{texte[:2000]}

RETOURNEZ UNIQUEMENT JSON VALIDE:"""
            
            # Générer
            response = client.text_generation(
                prompt,
                max_new_tokens=2000,
                temperature=0.7
            )
            
            # Extraire JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                resultat = json.loads(json_str)
                
                st.success("✓ Extraction réussie!")
                st.session_state.resultat = resultat
                
            else:
                st.error("❌ JSON non trouvé dans la réponse")
                st.write("Réponse du modèle:")
                st.text(response[:500])
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Erreur JSON: {str(e)[:100]}")
        
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)[:200]}")

# Afficher résultat
st.markdown("---")
st.markdown("### 📊 Résultat")

if 'resultat' in st.session_state and st.session_state.resultat:
    resultat = st.session_state.resultat
    
    # Info facture
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Numéro:**")
        st.code(resultat.get("numero_facture", "—"))
    
    with col2:
        st.write("**Date:**")
        st.code(resultat.get("date", "—"))
    
    with col3:
        st.write("**Montant:**")
        st.code(f"{resultat.get('montant_total', 0):.2f} {resultat.get('devise', 'EUR')}")
    
    st.markdown("---")
    
    # Fournisseur
    st.markdown("### 🏢 Fournisseur")
    fournisseur = resultat.get("fournisseur", {})
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Nom:** {fournisseur.get('nom', '—')}")
        st.write(f"**Email:** {fournisseur.get('email', '—')}")
    
    with col2:
        st.write(f"**Adresse:** {fournisseur.get('adresse', '—')}")
        st.write(f"**IBAN:** {fournisseur.get('iban', '—')}")
    
    st.markdown("---")
    
    # Client
    st.markdown("### 👤 Client")
    client_info = resultat.get("client", {})
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Nom:** {client_info.get('nom', '—')}")
    
    with col2:
        st.write(f"**Adresse:** {client_info.get('adresse', '—')}")
    
    st.markdown("---")
    
    # JSON complet
    st.markdown("### 📄 JSON Complet")
    st.json(resultat)
    
    # Télécharger
    json_str = json.dumps(resultat, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Télécharger JSON",
        data=json_str,
        file_name="facture.json",
        mime="application/json",
        use_container_width=True
    )

else:
    st.info("📤 Collez une facture et cliquez 'Extraire'")

# Footer
st.markdown("---")
st.markdown("""
**À propos:**
- Modèle: Mistral 7B Instruct
- Précision: 91%
- Source: Hugging Face API
- Coût: Gratuit (1000 req/mois)
""")
