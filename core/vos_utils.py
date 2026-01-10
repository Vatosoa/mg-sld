# vos_utils.py
import os
from django.conf import settings
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from .postag_utils import tokenize_words_pos

# td = Tso-drafitra
# md = Mivadi-drafitra
# Construct the path relative to your project root
stopwords_file_path_td = os.path.join(settings.BASE_DIR, 'data', 'corpus', 'stopwords-td.txt')

# Open the file using this dynamic path
with open(stopwords_file_path_td, 'r', encoding='utf-8') as file:
    stop_words_td = set(file.read().splitlines())

# Construct the path relative to your project root
stopwords_file_path_md = os.path.join(settings.BASE_DIR, 'data', 'corpus', 'stopwords-md.txt')

# Open the file using this dynamic path
with open(stopwords_file_path_md, 'r', encoding='utf-8') as file:
    stop_words_md = set(file.read().splitlines())

# Fonction de tokenisation de mots
def tokenize_words_td(text):
    if isinstance(text, str):
        text = re.sub(r'[-;:()?!,.\'"\/|]', ' ', text)
        tokens = nltk.word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words_td]
        return tokens
    else:
        return []
    

def tokenize_words_md(text):
    if isinstance(text, str):
        text = re.sub(r'[-;:()?!,.\'"\/|]', ' ', text)
        tokens = nltk.word_tokenize(text)
        #tokens = [token.lower() for token in tokens if token.isalnum()]
        tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words_md]

        return tokens
    else:
        return []
    

def tokenize_words_phrase(text):
    if isinstance(text, str):
        tokens = nltk.word_tokenize(text)
        tokens_with_special_chars = []
        for token in tokens:
            # Utiliser une expression régulière pour séparer les caractères spéciaux
            tokens_with_special_chars.extend(re.findall(r"[\w]+|[.,!?;']", token))
        return tokens_with_special_chars
    else:
        return []



def align_tokens_with_original(tokens_md, tokens_original):
    new_tokens = []

    for token_md in tokens_md:
        found = False
        for token_original in tokens_original:
            if token_md == token_original.lower():
                # Trouver l'emplacement de l'apostrophe dans le token original
                apostrophe_index = token_original.find("'")
                # Si l'apostrophe est présente, l'ajouter à la position correspondante dans le nouveau token
                if apostrophe_index != -1:
                    new_tokens[-1] = new_tokens[-1][:apostrophe_index] + "'" + new_tokens[-1][apostrophe_index:]
                else:
                    new_tokens.append(token_original)
                found = True
                break

        if not found:
            new_tokens.append(token_md)

    return new_tokens


def load_structure_vos_model():
    # 1. Définition des chemins dynamiques pour les modèles
    path_model_td = os.path.join(settings.BASE_DIR, 'data', 'models', 'model-mg-structurevos-td.h5')
    path_model_md = os.path.join(settings.BASE_DIR, 'data', 'models', 'model-mg-structurevos-md.h5')

    # 2. Définition des chemins pour les encodeurs
    base_pretrain = os.path.join(settings.BASE_DIR, 'data', 'pretraining', 'mg-structurevos')
    
    path_token_td = os.path.join(base_pretrain, 'token_encoder-td.joblib')
    path_token_md = os.path.join(base_pretrain, 'token_encoder-md.joblib')
    path_label_td = os.path.join(base_pretrain, 'label_encoder-td.joblib')
    path_label_md = os.path.join(base_pretrain, 'label_encoder-md.joblib')

    # 3. Chargement effectif des fichiers
    model_td = load_model(path_model_td)
    model_md = load_model(path_model_md)
    
    token_encoder_td = joblib.load(path_token_td)
    token_encoder_md = joblib.load(path_token_md)
    label_encoder_td = joblib.load(path_label_td)
    label_encoder_md = joblib.load(path_label_md)
    
    return model_td, model_md, token_encoder_td, token_encoder_md, label_encoder_td, label_encoder_md


def predict_vos_structure(model_td, model_md, token_encoder_td, token_encoder_md, label_encoder_td, label_encoder_md, predictions_pos, sentence):
    tokens_pos = tokenize_words_pos(sentence)
    
    if "dia" in tokens_pos and "Kianteny" in predictions_pos:
        structure_prediction = "Mivadi-drafitra"
        tokens_phrase_md = tokenize_words_md(sentence)  # Tokeniser la nouvelle phrase
        predictions_tokens_md = []  # Initialiser les prédictions pour cette phrase
        
        for token_md in tokens_phrase_md:
            token_encoded_md = token_encoder_md.transform([token_md])[0]  # Encoder le token
            token_padded_md = pad_sequences([[token_encoded_md]], maxlen=1, padding='post')  # Rembourrer le token
            prediction_token_md = model_md.predict(token_padded_md)  # Prédire l'étiquette pour le token
            decoded_token_md = label_encoder_md.inverse_transform(np.argmax(prediction_token_md, axis=-1))
            predictions_tokens_md.append(decoded_token_md[0])  # Ajouter l'étiquette prédite aux prédictions de la phrase

        new_tokens = align_tokens_with_original(tokens_phrase_md, tokenize_words_phrase(sentence))

        return predictions_tokens_md, structure_prediction, new_tokens
     
    else:
        structure_prediction = "Tso-drafitra"
        tokens_phrase_td = tokenize_words_td(sentence)  # Tokeniser la nouvelle phrase
        predictions_tokens_td = []  # Initialiser les prédictions pour cette phrase
        
        for token_td in tokens_phrase_td:
            token_encoded_td = token_encoder_td.transform([token_td])[0]  # Encoder le token
            token_padded_td = pad_sequences([[token_encoded_td]], maxlen=1, padding='post')  # Rembourrer le token
            prediction_token_td = model_td.predict(token_padded_td)  # Prédire l'étiquette pour le token
            decoded_token_td = label_encoder_td.inverse_transform(np.argmax(prediction_token_td, axis=-1))
            predictions_tokens_td.append(decoded_token_td[0])  # Ajouter l'étiquette prédite aux prédictions de la phrase

        new_tokens = align_tokens_with_original(tokens_phrase_td, tokenize_words_phrase(sentence))

        return predictions_tokens_td, structure_prediction, new_tokens
    