# postag_utils.py
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def tokenize_words_pos(text):
    if isinstance(text, str):
        text = re.sub(r'[-;:()?!,.\'"\/|]', ' ', text)
        tokens = nltk.word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalnum()]
        return tokens
    else:
        return []


def load_pos_tagging_model():
    # Chargement du modèle de POS tagging
    model = load_model('/home/vatosoa/mg-smart-lingua-discover/data/models/model-mg-postag.h5')
    
    # Chargement des encodeurs
    token_encoder = joblib.load('/home/vatosoa/mg-smart-lingua-discover/data/pretraining/mg-postag/token_encoder.joblib')
    label_encoder = joblib.load('/home/vatosoa/mg-smart-lingua-discover/data/pretraining/mg-postag/label_encoder.joblib')    
    return model, token_encoder, label_encoder


def predict_pos_tags(model, token_encoder, label_encoder, sentence):
    tokens = tokenize_words_pos(sentence)
    predictions_tokens = []
    
    for i, token in enumerate(tokens):
        token_encoded = token_encoder.transform([token])[0]
        token_padded = pad_sequences([[token_encoded]], maxlen=1, padding='post')
        prediction_token = model.predict(token_padded)
        decoded_token = label_encoder.inverse_transform(np.argmax(prediction_token, axis=-1))

        # condition pour "dia"
        if token.lower() == 'dia':
            if i == 0 or i >= 2:
                predictions_tokens.append('Mpampitohy')
            else:
                predictions_tokens.append('Kianteny')
        else:
            predictions_tokens.append(decoded_token[0])    

    return predictions_tokens
