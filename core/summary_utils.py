# summary_utils.py
import joblib
import nltk
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

max_summary_len = 100

stopwords_file_path = '/home/vatosoa/mg-smart-lingua-discover/data/corpus/stopwords-td.txt'
with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    stop_words = set(file.read().splitlines())

def text_cleaner(text, stopwords):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\([^)]*\)', '', text)  # Remove text between parentheses
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords])  # Remove stopwords
    return text.lower().strip()


def decode_sequence(encoder_model, decoder_model, input_seq, y_tokenizer, reverse_target_word_index, max_summary_len):
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = y_tokenizer.word_index['sostok']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        if sampled_token_index not in reverse_target_word_index:
            stop_condition = True
        elif sampled_token_index == y_tokenizer.word_index['eostok'] or len(decoded_sentence.split()) >= (max_summary_len - 1):
            stop_condition = True
        else:
            sampled_token = reverse_target_word_index[sampled_token_index]
            decoded_sentence += ' ' + sampled_token

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence.strip()




def seq2summary(input_seq, reverse_target_word_index):
    new_string = ''
    for i in input_seq:
        if i != 0:
            new_string += reverse_target_word_index[i] + ' '
    return new_string

def seq2text(input_seq, reverse_source_word_index):
    new_string = ''
    for i in input_seq:
        if i != 0:
            new_string += reverse_source_word_index[i] + ' '
    return new_string


def load_summary_resources():
    load_model_summary = '/home/vatosoa/mg-smart-lingua-discover/data/models/model-mg-summaryfki.h5'
    encoder_model_path = '/home/vatosoa/mg-smart-lingua-discover/data/models/encoder_mode-mg-summaryfki.h5'
    decoder_model_path = '/home/vatosoa/mg-smart-lingua-discover/data/models/decoder_model-mg-summaryfki.h5'
    x_tokenizer_path = '/home/vatosoa/mg-smart-lingua-discover/data/pretraining/mg-summaryfki/x_tokenizer.joblib'
    y_tokenizer_path = '/home/vatosoa/mg-smart-lingua-discover/data/pretraining/mg-summaryfki/y_tokenizer.joblib'
    reverse_target_word_index_path = '/home/vatosoa/mg-smart-lingua-discover/data/functions/reverse_target_word_index.joblib'

    loaded_model = load_model(load_model_summary)
    loaded_encoder_model = load_model(encoder_model_path)
    loaded_decoder_model = load_model(decoder_model_path)
    
    x_tokenizer = joblib.load(x_tokenizer_path)
    y_tokenizer = joblib.load(y_tokenizer_path)
    reverse_target_word_index = joblib.load(reverse_target_word_index_path)

    return loaded_encoder_model, loaded_decoder_model, x_tokenizer, y_tokenizer, reverse_target_word_index, loaded_model



def predict_summary(x_tokenizer, y_tokenizer, max_summary_len, encoder_model, decoder_model, reverse_target_word_index, sentence):
    cleaned_input_text = text_cleaner(sentence, stop_words)
    input_sequence = x_tokenizer.texts_to_sequences([cleaned_input_text])[0]
    
    # forme de la séquence d'entrée
    print("Input sequence shape:", len(input_sequence))

    input_sequence_padded = pad_sequences([input_sequence], maxlen=max_summary_len, padding='post')
    print("Padded input sequence shape:", input_sequence_padded.shape)

    predictions_summary = decode_sequence(encoder_model, decoder_model, input_sequence_padded, y_tokenizer, reverse_target_word_index, max_summary_len)
    return predictions_summary
