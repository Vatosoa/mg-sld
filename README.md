# MG-SLD (Malagasy Smart Lingua Discover)

MG-SLD is a research project specializing in natural language processing (NLP) for the Malagasy language. It enables morphosyntactic tagging (**Part-of-Speech tagging**) and in-depth analysis of sentence structures.

## 🛠 Installation and Configuration

This project is optimized for **Python 3.11**, ensuring stable compatibility with **TensorFlow 2.15/2.16** and **Keras**.

### 1. Preparing the Python environment

In 2026, **Python 3.11** is no longer the standard version of the system, so it is recommended to use `pyenv` for isolated management:

```bash
# Install the specific version
pyenv install 3.11.7

# Set the local version for this folder
pyenv local 3.11.7
```

### 2. Creating and activating the virtual environment
```bash
# Creating the environment
python -m venv venv-3.11.7

# Activation (Linux/macOS)
source venv-3.11.7/bin/activate

# Activation (Windows)
venv-3.11.7\Scripts\activate
```

### 3. Installing dependencies
```bash
# Updating pip
pip install --upgrade pip

# Installing dependencies
pip install -r requirements.txt

# In case of network issues
pip install --default-timeout=10000 -r requirements.txt
```

### 4. Configuring NLTK resources
The project uses the NLTK library for word segmentation. You must install the following data models after activating your environment:

Manual method (via the Python interpreter):
```bash
import nltk
nltk.download(‘punkt’)
nltk.download(‘punkt_tab’)
```

## 🚀 Usage
To launch the Django application locally:

### 1. Apply migrations (if necessary)
```bash
python manage.py migrate
```

### 2. Start the server
```bash
python manage.py runserver
```

### 3. Access the interface
Go to **http://127.0.0.1:8000** in your browser.

#### 📂 Project structure

```bash
MG-SLD/
├── core/                 # Contains the main logic (views, URLs)
├── data/
│   ├── corpus/
│   ├── functions/
│   ├── models/          # Trained models (.h5)
│   └── pretraining/     # Contains encoders (.joblib)
├── notebooks/
├── static/              # CSS files, images, and visual resources
├── staticfiles/         
├── staticfiles_build/ 
├── templates/         
│   ├── postag.html
├── vev-3.11.7
├── manage.py
├── README.md
└── requirements.txt     # Project dependencies
```

# ✍️ Author
Vatosoa Razafindrazaka 

## ⭐ Feel free to contribute or report issues via Issues!
