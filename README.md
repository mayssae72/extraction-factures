# 📄 Extraction de Factures avec Mistral 7B

> Application NLP qui extrait automatiquement les données structurées de factures textuelles via Mistral 7B (Hugging Face API) — sans installation de modèle local.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Mistral](https://img.shields.io/badge/Mistral-7B_Instruct-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Fonctionnalités

- 📥 **Input** : texte brut de facture (copier-coller)
- 🤖 **Modèle** : Mistral 7B Instruct via Hugging Face Inference API
- 📤 **Output** : JSON structuré avec tous les champs extraits
- 💾 **Export** : téléchargement du JSON en un clic
- ☁️ **100% cloud** : aucun téléchargement de modèle requis

---

## 📊 Champs extraits

```json
{
  "numero_facture": "FAC-2024-001",
  "date": "15/01/2024",
  "montant_total": 1500.00,
  "devise": "EUR",
  "fournisseur": {
    "nom": "Acme SARL",
    "adresse": "12 rue de Paris, 75001 Paris",
    "email": "contact@acme.fr",
    "iban": "FR76..."
  },
  "client": {
    "nom": "GlobalRetail Inc",
    "adresse": "45 avenue des Champs, 75008 Paris"
  }
}
```

---

## 🚀 Installation & Lancement

### 1. Cloner le repo
```bash
git clone https://github.com/mayssae72/extraction-factures.git
cd extraction-factures
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'application
```bash
streamlit run app.py
```

### 5. Configurer le token Hugging Face
- Créer un token gratuit sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Le coller dans la sidebar de l'application

---

## 🛠️ Stack technique

| Outil | Rôle |
|-------|------|
| `Mistral 7B Instruct` | Modèle LLM pour l'extraction |
| `Hugging Face API` | Inférence cloud gratuite |
| `Streamlit` | Interface web interactive |
| `Python 3.9+` | Langage principal |

---

## 📁 Structure du projet

```
extraction-factures/
├── app.py                  # Application principale Streamlit
├── requirements.txt        # Dépendances Python
├── .streamlit/             # Configuration Streamlit
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

Le token Hugging Face est requis pour utiliser l'API d'inférence Mistral 7B.

1. Créer un compte gratuit sur [huggingface.co](https://huggingface.co)
2. Générer un token sur [Settings → Tokens](https://huggingface.co/settings/tokens)
3. Le coller dans la sidebar de l'app au lancement

> ⚠️ **Note réseau Maroc** : si `raw.githubusercontent.com` est bloqué, utilisez un VPN (ProtonVPN ou Windscribe sont gratuits).

---

## 👩‍💻 Auteure

**Mayssae Atifi**
Master Data Science & Intelligence Artificielle

[![GitHub](https://img.shields.io/badge/GitHub-mayssae72-black)](https://github.com/mayssae72)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mayssae--atifi-blue)](https://linkedin.com/in/mayssae-atifi)

---

## 📄 Licence

MIT License — voir [LICENSE](LICENSE)
