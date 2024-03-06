# views.py
from django.shortcuts import render
from django.http import HttpResponse
from .postag_utils import load_pos_tagging_model, predict_pos_tags, tokenize_words_pos
from .vos_utils import load_structure_vos_model, predict_vos_structure, tokenize_words_md, tokenize_words_td
from .summary_utils import load_summary_resources, predict_summary
from keras.preprocessing.sequence import pad_sequences
import joblib



def postag_vos_view(request):
    user_sentence = ""
    predictions_pos = []
    predictions_vos = []
    tokenize_vos = []
    predictions_summary = ""
    structure_prediction = ""
    error_message = ""
    sentence_multiple_error = ""

    if request.method == 'POST':
        user_sentence = request.POST.get('user_sentence', '')

        if user_sentence:
            # senctence count
            count_periods = user_sentence.count('.')
            if count_periods > 1:
                sentence_multiple_error = "Please enter only one sentence."
            else:
                try:    
                    # POS Tagging
                    loaded_model_pos, loaded_token_encoder_pos, loaded_label_encoder_pos = load_pos_tagging_model()
                    predictions_pos = predict_pos_tags(loaded_model_pos, loaded_token_encoder_pos, loaded_label_encoder_pos, user_sentence)
                    tokenize_pos = tokenize_words_pos(user_sentence)
                    predictions_pos_formatted = ' '.join([f"{token}/{pos}" for token, pos in zip(tokenize_pos, predictions_pos)])


                    # VOS/EFL Structure
                    model_td, model_md, token_encoder_td, token_encoder_md, label_encoder_td, label_encoder_md = load_structure_vos_model()
                    predictions_vos, structure_prediction, new_tokens = predict_vos_structure(model_td, model_md, token_encoder_td, token_encoder_md, label_encoder_td, label_encoder_md, predictions_pos, user_sentence)
                    # pedictions_vos sentence
                    #predictions_vos = ' '.join(predictions_vos)
                    # structure_prediction
                    if structure_prediction == "Mivadi-drafitra":
                        tokenize_vos = tokenize_words_md(user_sentence)
                        predictions_vos_formatted = ' '.join([f"{token}/{vos}" for token, vos in zip(tokenize_words_md(user_sentence), predictions_vos)])
                    else:
                        tokenize_vos = tokenize_words_td(user_sentence)
                        predictions_vos_formatted = ' '.join([f"{token}/{vos}" for token, vos in zip(tokenize_words_td(user_sentence), predictions_vos)])
                    # new_tokens
                    fki = ' '.join(new_tokens) + '.'


                    # Summary Prediction
                    loaded_encoder_model, loaded_decoder_model, x_tokenizer, y_tokenizer, reverse_target_word_index, _ = load_summary_resources()
                    max_summary_len = 100
                    predictions_summary = predict_summary(x_tokenizer, y_tokenizer, max_summary_len, loaded_encoder_model, loaded_decoder_model, reverse_target_word_index, user_sentence)

                    context = {
                    'user_sentence': user_sentence, 
                    'predictions_pos_formatted': predictions_pos_formatted, 
                    'predictions_vos_formatted': predictions_vos_formatted, 
                    'fki': fki, 
                    'predictions_summary': predictions_summary, 
                    'structure_prediction': structure_prediction,
                    'error_message': error_message,
                    'sentence_multiple_error': sentence_multiple_error,
                    }

                    return render(request, 'postag.html', context)

                
                except ValueError as e:
                    error_message = "Mialatsiny, some words in your input are not recognized in our database. Feel free to contact us via our Facebook/Twitter account for further assistance: Vatosoa Razafindrazaka"
                    context = {'error_message': error_message}
                    return render(request, 'postag.html', context)
                
        else:
            return HttpResponse("Please add a sentence")

    return render(request, 'postag.html')
