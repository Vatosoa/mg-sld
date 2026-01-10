# MG-SLD (Malagasy Smart Lingua Discover)

MG-SLD est un projet de recherche spécialisé dans le traitement du langage naturel (NLP) pour la langue malgache. Il permet notamment l'étiquetage morphosyntaxique (**Part-of-Speech tagging**) et l'analyse approfondie des structures de phrases.

## 🛠 Installation et Configuration

Ce projet est optimisé pour **Python 3.11**, garantissant une compatibilité stable avec **TensorFlow 2.15/2.16** et **Keras**.

### 1. Préparation de l'environnement Python

En 2026, **Python 3.11** n'étant plus la version standard du système, il est recommandé d'utiliser `pyenv` pour une gestion isolée :

```bash
# Installer la version spécifique
pyenv install 3.11.7

# Définir la version locale pour ce dossier
pyenv local 3.11.7

```

### 2. Création et activation de l'environnement virtuel
```bash
# Création de l'environnement
python -m venv venv-3.11.7

# Activation (Linux/macOS)
source venv-3.11.7/bin/activate

# Activation (Windows)
venv-3.11.7\Scripts\activate

```

### 3. Installation des dépendances
```bash
# Mise à jour de pip
pip install --upgrade pip

# Installation des dépendances
pip install -r requirements.txt

# En cas de problèmes de réseau
pip install --default-timeout=10000 -r requirements.txt

```

### 4. Configuration des ressources NLTK
Le projet utilise la bibliothèque NLTK pour la segmentation des mots. Vous devez impérativement installer les modèles de données suivants après avoir activé votre environnement :

Méthode manuelle (via l'interpréteur Python) :
```bash
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 🚀 Utilisation
Pour lancer l'application Django localement :

### 1. Appliquez les migrations (si nécessaire)
```bash
python manage.py migrate
```

### 2. Démarrez le serveur
```bash
python manage.py runserver
```

### 3. Accédez à l'interface
Rendez-vous sur **http://127.0.0.1:8000** dans votre navigateur.

#### 📂 Structure du projet

```bash
MG-SLD/
├── core/                 # Contient la logique principale (views, urls)
├── data/
│   ├── models/          # Modèles entraînés (.h5)
│   └── pretraining/     # Contient les encodeurs (.joblib)
├── static/              # Fichiers CSS, images et ressources visuelles
├── requirements.txt     # Dépendances du projet
└── manage.py           # Script de gestion Django
```

# ✍️ Auteur
Vatosoa Razafindrazaka 

## ⭐ N'hésitez pas à contribuer ou à signaler des problèmes via les Issues !
