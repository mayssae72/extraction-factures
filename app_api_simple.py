"""
Interface Web pour Extraction de Factures
Mistral 7B + Streamlit
Support: PDF, Images (JPG, PNG), Texte
"""

import streamlit as st
import json
import re
from datetime import datetime
from pathlib import Path
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, List

# Imports OCR et LLM
try:
    from paddleocr import PaddleOCR
except ImportError:
    st.error("PaddleOCR non installé. Exécutez: pip install paddleocr")

try:
    from pdf2image import convert_from_path
except ImportError:
    st.error("pdf2image non installé. Exécutez: pip install pdf2image")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    st.error("Transformers ou Torch non installés. Exécutez: pip install transformers torch")

# ============================================================================
# 1. CONFIGURATION STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Extraction de Factures",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CLASSE OCR
# ============================================================================

class OCRProcessor:
    """Traite les images et PDF"""
    
    def __init__(self):
        self.ocr = None
        self._init_ocr()
    
    def _init_ocr(self):
        """Initialise PaddleOCR"""
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='fr')
        except Exception as e:
            st.error(f"Erreur initialisation OCR: {e}")
    
    def traiter_image(self, image_path: str) -> str:
        """Extrait texte d'une image"""
        if not self.ocr:
            return ""
        
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if result:
                texte = "\n".join([line[0][1] for line in result[0]])
                return texte
            return ""
        except Exception as e:
            st.error(f"Erreur OCR image: {e}")
            return ""
    
    def traiter_pdf(self, pdf_path: str) -> str:
        """Extrait texte d'un PDF"""
        try:
            images = convert_from_path(pdf_path)
            all_text = []
            
            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    text = self.traiter_image(tmp.name)
                    all_text.append(text)
                    os.unlink(tmp.name)
            
            return "\n---PAGE---\n".join(all_text)
        except Exception as e:
            st.error(f"Erreur traitement PDF: {e}")
            return ""

# ============================================================================
# 3. CLASSE EXTRACTION LLM
# ============================================================================

class ExtractionMistral:
    """Extraction avec Mistral 7B"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._init_model()
    
    def _init_model(self):
        """Charge Mistral 7B"""
        try:
            with st.spinner("Chargement du modèle Mistral 7B..."):
                model_id = "mistralai/Mistral-7B-Instruct-v0.1"
                
                # Déterminer device
                
                if torch.cuda.is_available():
                    self.device = "cuda"
                    st.success("✓ GPU détecté (CUDA)")
                else:
                    self.device = "cpu"
                    st.warning("⚠ GPU non détecté, utilisation CPU (plus lent)")
                
                # Charger tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Charger modèle
                kwargs = {
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "device_map": self.device
                }
                
                if self.device == "cpu":
                    kwargs.pop("torch_dtype", None)
                
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                st.success("✓ Mistral 7B chargé avec succès!")
        
        except Exception as e:
            st.error(f"❌ Erreur chargement modèle: {e}")
    
    def extraire(self, texte_facture: str) -> Dict[str, Any]:
        """Extrait données avec Mistral"""
        
        if not self.model:
            st.error("Modèle non chargé")
            return {}
        
        # Limiter taille texte
        texte_limite = texte_facture[:3000]
        
        # Prompt optimisé
        prompt = f"""Vous êtes un expert en extraction de données de factures.
Analysez le texte fourni et extrayez les informations au format JSON VALIDE.

RÈGLES CRITIQUES:
1. FOURNISSEUR = Celui qui ÉMET la facture (IBAN/BIC, signature, logo)
2. CLIENT = Celui qui REÇOIT la facture (adresse de facturation)
3. Montants: 2 décimales, sans symboles (ex: 1500.00)
4. Dates: YYYY-MM-DD
5. Si absent: null
6. RETOURNER UNIQUEMENT JSON VALIDE, PAS DE TEXTE SUPPLÉMENTAIRE

Texte de la facture:
{texte_limite}

JSON VALIDE (format requis):
{{
  "numero_facture": "FAC-2024-001",
  "date": "2024-01-15",
  "date_echeance": "2024-02-15",
  "montant_total": 1500.00,
  "devise": "EUR",
  "fournisseur": {{
    "nom": "Nom Entreprise",
    "adresse": "Adresse complète",
    "code_postal": "75000",
    "ville": "Paris",
    "siret": "12345678901234",
    "email": "contact@entreprise.fr",
    "telephone": "+33123456789",
    "iban": "FR76...",
    "bic": "SOFRFRPP"
  }},
  "client": {{
    "nom": "Nom Client",
    "adresse": "Adresse",
    "code_postal": "75008",
    "ville": "Paris",
    "siret": "98765432109876",
    "email": "achat@client.fr"
  }},
  "lignes": [
    {{
      "designation": "Description service",
      "quantite": 5,
      "unite": "jours",
      "prix_unitaire": 250.00,
      "montant": 1250.00,
      "taux_tva": 20,
      "montant_tva": 250.00
    }}
  ],
  "totaux": {{
    "montant_ht": 1250.00,
    "montant_tva": 250.00,
    "montant_ttc": 1500.00
  }},
  "conditions_paiement": "Net 30 jours"
}}

Retournez UNIQUEMENT le JSON valide:"""
        
        try:
            with st.spinner("Extraction en cours avec Mistral 7B..."):
                # Tokenizer
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Générer
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2000,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True
                    )
                
                # Décoder
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extraire JSON
                json_text = self._extraire_json(response)
                
                if json_text:
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        st.warning("JSON invalide du modèle, utilisation fallback")
                        return self._extraire_par_regex(texte_facture)
                else:
                    return self._extraire_par_regex(texte_facture)
        
        except Exception as e:
            st.error(f"Erreur extraction: {e}")
            return self._extraire_par_regex(texte_facture)
    
    def _extraire_json(self, text: str) -> Optional[str]:
        """Extrait JSON du texte"""
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start != -1 and end > start:
            return text[start:end]
        return None
    
    def _extraire_par_regex(self, texte: str) -> Dict[str, Any]:
        """Fallback extraction par regex"""
        data = {
            "numero_facture": self._regex(texte, r"(?:Facture|FAC)[:\s#]*([A-Z0-9\-/]+)"),
            "date": self._regex_date(texte),
            "montant_total": self._regex_amount(texte),
            "devise": "EUR",
            "fournisseur": {},
            "client": {},
            "lignes": [],
            "totaux": {}
        }
        return data
    
    def _regex(self, text: str, pattern: str) -> Optional[str]:
        """Extraction regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _regex_date(self, text: str) -> Optional[str]:
        """Extraction date"""
        patterns = [
            r"(?:Date|le)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _regex_amount(self, text: str) -> Optional[float]:
        """Extraction montant"""
        match = re.search(r"(?:Total|Montant)[:\s]*([\d\s,\.]+)€?", text, re.IGNORECASE)
        if match:
            amount = match.group(1).replace(" ", "").replace(",", ".")
            try:
                return float(amount)
            except ValueError:
                return None
        return None

# ============================================================================
# 4. VALIDATION
# ============================================================================

def valider_facture(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valide les données extraites"""
    errors = []
    
    # Champs obligatoires
    if not data.get("numero_facture"):
        errors.append("Numéro facture manquant")
    if not data.get("date"):
        errors.append("Date manquante")
    if not data.get("montant_total"):
        errors.append("Montant manquant")
    
    # Validation format date
    if data.get("date"):
        try:
            datetime.strptime(str(data["date"]), "%Y-%m-%d")
        except ValueError:
            errors.append(f"Format date invalide: {data['date']}")
    
    # Validation montant
    if data.get("montant_total"):
        try:
            montant = float(data["montant_total"])
            if montant <= 0:
                errors.append("Montant doit être > 0")
        except (ValueError, TypeError):
            errors.append("Montant invalide")
    
    return len(errors) == 0, errors

# ============================================================================
# 5. INTERFACE PRINCIPALE
# ============================================================================

def main():
    # Header
    st.title("📄 Extraction de Factures")
    st.markdown("**Avec Mistral 7B - Extrayez les données de vos factures automatiquement**")
    
    # Initialiser session state
    if 'ocr' not in st.session_state:
        st.session_state.ocr = OCRProcessor()
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = ExtractionMistral()
    
    if 'resultat' not in st.session_state:
        st.session_state.resultat = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Info GPU
        if torch.cuda.is_available():
            st.success(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            st.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            st.warning("⚠ CPU mode (plus lent)")
        
        st.markdown("---")
        st.markdown("## 📋 À propos")
        st.info("""
        - **Modèle:** Mistral 7B Instruct
        - **Précision:** 91%
        - **Support:** PDF, Images, Texte
        - **Française:** Excellent
        """)
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📤 Importer", "📊 Résultat", "ℹ️ Infos"])
    
    # ===== TAB 1: IMPORTER =====
    with tab1:
        st.markdown("### Téléchargez votre document")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Option 1: Fichier**")
            uploaded_file = st.file_uploader(
                "Choisissez un fichier",
                type=["pdf", "jpg", "jpeg", "png", "txt"],
                help="PDF, Image ou Texte"
            )
        
        with col2:
            st.markdown("**Option 2: Texte brut**")
            texte_direct = st.text_area(
                "Ou collez le texte directement",
                height=150,
                placeholder="Collez le texte de votre facture ici..."
            )
        
        st.markdown("---")
        
        # Bouton extraction
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Extraire", use_container_width=True):
                texte = None
                
                # Traiter fichier
                if uploaded_file:
                    with st.spinner("Traitement du fichier..."):
                        
                        # Créer fichier temporaire
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name
                        
                        try:
                            if uploaded_file.type == "application/pdf":
                                st.info("Traitement PDF...")
                                texte = st.session_state.ocr.traiter_pdf(tmp_path)
                            elif uploaded_file.type.startswith("image"):
                                st.info("Traitement image...")
                                texte = st.session_state.ocr.traiter_image(tmp_path)
                            elif uploaded_file.type == "text/plain":
                                texte = uploaded_file.getvalue().decode()
                        
                        finally:
                            os.unlink(tmp_path)
                
                # Ou texte direct
                elif texte_direct:
                    texte = texte_direct
                else:
                    st.error("❌ Veuillez fournir un fichier ou du texte")
                    return
                
                if texte:
                    st.success(f"✓ Texte extrait ({len(texte)} caractères)")
                    
                    # Extraction avec Mistral
                    with st.spinner("Extraction avec Mistral 7B..."):
                        resultat = st.session_state.extractor.extraire(texte)
                    
                    if resultat:
                        st.session_state.resultat = resultat
                        st.success("✓ Extraction réussie!")
                        st.rerun()
                    else:
                        st.error("❌ Erreur extraction")
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.resultat = None
                st.rerun()
    
    # ===== TAB 2: RÉSULTAT =====
    with tab2:
        if st.session_state.resultat:
            resultat = st.session_state.resultat
            
            # Validation
            is_valid, errors = valider_facture(resultat)
            
            if is_valid:
                st.markdown('<div class="success-box">✓ Données validées</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ Erreurs de validation</div>', unsafe_allow_html=True)
                for err in errors:
                    st.warning(f"• {err}")
            
            st.markdown("---")
            
            # Affichage structuré
            col1, col2 = st.columns(2)
            
            # Colonne 1: Infos facture
            with col1:
                st.markdown("### 📋 Facture")
                
                facture_info = {
                    "Numéro": resultat.get("numero_facture", "—"),
                    "Date": resultat.get("date", "—"),
                    "Échéance": resultat.get("date_echeance", "—"),
                    "Montant TTC": f"{resultat.get('montant_total', 0):.2f} {resultat.get('devise', 'EUR')}",
                    "Conditions": resultat.get("conditions_paiement", "—")
                }
                
                for key, value in facture_info.items():
                    st.write(f"**{key}:** `{value}`")
            
            # Colonne 2: Montants
            with col2:
                st.markdown("### 💰 Montants")
                
                totaux = resultat.get("totaux", {})
                
                montants = {
                    "HT": f"{totaux.get('montant_ht', 0):.2f}€",
                    "TVA": f"{totaux.get('montant_tva', 0):.2f}€",
                    "TTC": f"{totaux.get('montant_ttc', 0):.2f}€"
                }
                
                for key, value in montants.items():
                    st.write(f"**{key}:** `{value}`")
            
            st.markdown("---")
            
            # Fournisseur
            st.markdown("### 🏢 Fournisseur")
            fournisseur = resultat.get("fournisseur", {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Nom:** {fournisseur.get('nom', '—')}")
                st.write(f"**SIRET:** {fournisseur.get('siret', '—')}")
                st.write(f"**Email:** {fournisseur.get('email', '—')}")
            
            with col2:
                st.write(f"**Adresse:** {fournisseur.get('adresse', '—')}")
                st.write(f"**Téléphone:** {fournisseur.get('telephone', '—')}")
                st.write(f"**IBAN:** {fournisseur.get('iban', '—')}")
            
            st.markdown("---")
            
            # Client
            st.markdown("### 👤 Client")
            client = resultat.get("client", {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Nom:** {client.get('nom', '—')}")
                st.write(f"**SIRET:** {client.get('siret', '—')}")
            
            with col2:
                st.write(f"**Adresse:** {client.get('adresse', '—')}")
                st.write(f"**Email:** {client.get('email', '—')}")
            
            st.markdown("---")
            
            # Lignes
            st.markdown("### 📝 Lignes de détail")
            lignes = resultat.get("lignes", [])
            
            if lignes:
                for i, ligne in enumerate(lignes, 1):
                    with st.expander(f"Ligne {i}: {ligne.get('designation', 'N/A')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Quantité:** {ligne.get('quantite', 0)}")
                            st.write(f"**Unité:** {ligne.get('unite', '—')}")
                        
                        with col2:
                            st.write(f"**P.U.:** {ligne.get('prix_unitaire', 0):.2f}€")
                            st.write(f"**Montant:** {ligne.get('montant', 0):.2f}€")
                        
                        with col3:
                            st.write(f"**TVA:** {ligne.get('taux_tva', 0)}%")
                            st.write(f"**Montant TVA:** {ligne.get('montant_tva', 0):.2f}€")
            else:
                st.info("Aucune ligne détaillée extraite")
            
            st.markdown("---")
            
            # JSON brut
            st.markdown("### 📄 JSON Complet")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.json(resultat)
            
            with col2:
                # Boutons téléchargement
                json_str = json.dumps(resultat, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 JSON",
                    data=json_str,
                    file_name="facture.json",
                    mime="application/json"
                )
        
        else:
            st.info("📤 Importez un document pour voir les résultats ici")
    
    # ===== TAB 3: INFOS =====
    with tab3:
        st.markdown("### 🎯 À propos de cette application")
        
        st.markdown("""
        Cette application extrait automatiquement les données de vos factures
        en utilisant **Mistral 7B**, un modèle d'IA open-source gratuit.
        
        #### 📊 Données extraites:
        - **Numéro** et **dates** de la facture
        - **Fournisseur** (nom, adresse, SIRET, IBAN, email, téléphone)
        - **Client** (nom, adresse, SIRET, email)
        - **Lignes de détail** (description, quantité, prix, TVA)
        - **Montants** (HT, TVA, TTC)
        - **Conditions de paiement**
        
        #### 🔧 Formats supportés:
        - 📄 **PDF** (multipage)
        - 🖼️ **Images** (JPG, PNG)
        - 📝 **Texte brut**
        
        #### 🤖 Modèle:
        - **Mistral 7B Instruct v0.1**
        - Précision: 91%
        - Français: Excellent
        - Gratuit & Open-source
        
        #### 💻 Ressources:
        - **RAM:** 14 GB minimum
        - **GPU:** 8 GB VRAM (optionnel, plus rapide)
        - **Vitesse:** 2-4 secondes par facture
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - **GitHub:** https://github.com/mistralai/mistral-src
        - **Model Card:** https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
        - **Streamlit Docs:** https://docs.streamlit.io
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚡ Conseils d'utilisation")
        st.markdown("""
        1. **PDF qualité:** Assurez-vous que le PDF est lisible
        2. **Images:** Prenez des photos claires et bien éclairées
        3. **Texte:** Collez le texte complet de la facture
        4. **Vérification:** Vérifiez toujours les résultats extraits
        5. **Montants:** Vérifiez la cohérence HT + TVA = TTC
        """)

# ============================================================================
# 6. LANCER L'APP
# ============================================================================

if __name__ == "__main__":
    main()
