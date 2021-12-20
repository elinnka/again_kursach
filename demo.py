import pandas as pd
from textblob import TextBlob
import numpy as np
from nltk.tokenize import RegexpTokenizer
import benepar
from benepar.spacy_plugin import BeneparComponent
import spacy
import pickle
from models import Logreg as lg

tokenizer = RegexpTokenizer(r'\w+')
benepar.download('benepar_en3')
# Loading spaCy’s en model and adding benepar model to its pipeline
nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en3'))


def demo_preparing(demo_context, demo_question):
    # фильтрация слов
    demo_sentencesFiltred = (' '.join(
        map(str, [word for word in demo_context.split(" ") if word not in stopwords.words('english')])).lower())
    # деление на предложения
    blob = TextBlob(str(demo_sentencesFiltred))
    demo_sentencesInContextDiv = [item.raw for item in blob.sentences]
    # убираем пунктуацию
    demo_sentence_wtht_punct = []
    for j in range(len(demo_sentencesInContextDiv)):
        demo_sentence_wtht_punct.append(tokenizer.tokenize(demo_sentencesInContextDiv[j][:-1]))
    demo_sent_wtht_punct_inContext = demo_sentence_wtht_punct
    # делаем строки
    demo_sentences = []
    for sent in demo_sent_wtht_punct_inContext:
        demo_sentences.append(' '.join(map(str, sent)))
    demo_sentencesInContext = demo_sentences

    demo_questionsFiltred = (
        ' '.join([word.lower() for word in demo_question.split(" ") if word not in stopwords.words('english')]))
    blob = TextBlob(str(demo_questionsFiltred))
    demo_questionsInContextDiv = str([item.raw for item in blob.sentences])
    demo_questions_wtht_punct = tokenizer.tokenize(demo_questionsInContextDiv[:-1])
    demo_questions_prepared = str(demo_questions_wtht_punct).lower()

    return demo_sentencesInContext, demo_questions_prepared


def demo_data(demo_context, demo_question, demo_context_beg, demo_question_beg):
    demo_context, demo_question = demo_preparing(demo_context, demo_question)
    sents = []
    sentsInSentence = []
    for sentence in demo_context:
        sents_morethan1 = []
        if (sentence != ""):
            for el in list(nlp(sentence).sents):
                sents_morethan1.extend(list(el._.constituents))
        sentsInSentence.append(sents_morethan1)
    if len(sentsInSentence) == 0:
        sentsInSentence = [""]
    sents.append(sentsInSentence)

    root_sentenceInContext = []
    root_sentenceInContext_help = []
    sentence = demo_context_beg.replace("?", ".").replace(". ", ".").split(".")
    for j in range(len(sentence)):
        doc_sentencesInContext = nlp(sentence[j])
        root_sentence = []
        for el in doc_sentencesInContext.sents:
            root_sentence.append(el.root)
        root_sentenceInContext_help.append(root_sentence)
    demo_root_sentenceInContext = root_sentenceInContext_help

    demo_candidates = []
    for sent in sents[0]:
        demo_candidates.append(sent)

    doc_sentencesInContext = nlp(demo_question_beg)
    demo_root_sentence = []
    for el in doc_sentencesInContext.sents:
        demo_root_sentence.append(el.root)

    # 4
    length_span = []
    length_sentence = []
    length_left = []
    length_right = []
    # 1
    sum_tfidf_left = []
    sum_tfidf_right = []
    sum_tfidf_span = []
    sum_tfidf_sentence = []
    # 2
    sum_tfidf_left_bigrams = []
    sum_tfidf_right_bigrams = []
    sum_tfidf_span_bigrams = []
    sum_tfidf_sentence_bigrams = []
    # 3
    root_match = []
    # 5
    sum_tfidf_span_notapp = []

    cand_ans_df = []
    num_quest = 0
    num_sent_in_quest = 0
    numbers_of_quest = []
    numbers_of_sentence_in_quest = []
    num_sent_in_quest = 0
    for sentence in demo_candidates:
        for candidate in sentence:
            l_r = length_left_right(str(demo_context[num_sent_in_quest]).split(" "), str(candidate))
            length_left.append(l_r[0])
            length_right.append(l_r[1])
            length_span.append(len(str(candidate).split(" ")))
            length_sentence.append(len(demo_context[num_sent_in_quest].split(" ")))
            s_l_r_span = tfidf_sent_l_r_span(demo_context[num_sent_in_quest], demo_question, demo_context,
                                             l_r[0],
                                             len(str(candidate).split(" ")))

            s_l_r_span_bigrams = tfidf_bigrams_sent_l_r_span(demo_context[num_sent_in_quest],
                                                             demo_question, demo_context,
                                                             l_r[0],
                                                             len(str(candidate).split(" ")))

            sum_tfidf_sentence.append(s_l_r_span[0])
            sum_tfidf_left.append(s_l_r_span[1])
            sum_tfidf_right.append(s_l_r_span[2])
            sum_tfidf_span.append(s_l_r_span[3])

            sum_tfidf_sentence_bigrams.append(s_l_r_span_bigrams[0])
            sum_tfidf_left_bigrams.append(s_l_r_span_bigrams[1])
            sum_tfidf_right_bigrams.append(s_l_r_span_bigrams[2])
            sum_tfidf_span_bigrams.append(s_l_r_span_bigrams[3])

            sum_tfidf_span_notapp.append(tfidf_span_notapp(str(candidate),
                                                           demo_context[num_sent_in_quest],
                                                           demo_question, demo_context))
            if (num_sent_in_quest < len(demo_root_sentenceInContext[num_quest])):
                root_match.append(
                    int(match_roots(demo_root_sentenceInContext[num_sent_in_quest], demo_root_sentence[num_quest])))
            else:
                root_match.append(0)
            cand_ans_df.append(candidate)
            numbers_of_sentence_in_quest.append(num_sent_in_quest)
            numbers_of_quest.append(num_quest)
        num_sent_in_quest += 1

    demo_lengths = pd.DataFrame({'question_num': numbers_of_quest,
                                 'sentence_in_context_num': numbers_of_sentence_in_quest,
                                 'cand_ans': cand_ans_df,
                                 'root_match': root_match,
                                 'length_span': length_span,
                                 'length_sent': length_sentence, 'length_left': length_left,
                                 'length_right': length_right, 'tfidf_left': sum_tfidf_left,
                                 'tfidf_right': sum_tfidf_right, 'tfidf_span': sum_tfidf_span,
                                 'tfidf_sentence': sum_tfidf_sentence, 'span_word_freq': sum_tfidf_span_notapp,
                                 'tfidf_left_bigrams': sum_tfidf_left_bigrams,
                                 'tfidf_right_bigrams': sum_tfidf_right_bigrams,
                                 'tfidf_span_bigrams': sum_tfidf_span_bigrams,
                                 'tfidf_sentence_bigrams': sum_tfidf_sentence_bigrams})
    demo_test = np.array(
        [list(map(int, demo_lengths['root_match'])), demo_lengths['length_span'], demo_lengths['length_sent'],
         demo_lengths['length_left'], demo_lengths['length_right'], demo_lengths['tfidf_left'],
         demo_lengths['tfidf_right'], demo_lengths['tfidf_span'], demo_lengths['tfidf_sentence'],
         demo_lengths['span_word_freq'], demo_lengths['tfidf_left_bigrams'], demo_lengths['tfidf_right_bigrams'],
         demo_lengths['tfidf_span_bigrams'], demo_lengths['tfidf_sentence_bigrams']]).transpose()
    demo_text_test = np.array([demo_lengths['cand_ans'], demo_lengths['question_num']]).transpose()
    demo_test = (demo_test - np.mean(demo_test, axis=0)) / np.std(demo_test)
    return demo_test, demo_text_test


def demo_get_answer(demo_context, demo_question, demo_context_beg, demo_question_beg):
    file_path = 'weight.pickle'
    with open (file_path, 'rb') as f:
        new_data = pickle.load(f)
    w, b = new_data['w'], new_data['b']
    demo_test, demo_text_test = demo_data(demo_context, demo_question, demo_context_beg, demo_question_beg)
    demo_prediction = Predict(demo_test, w, b)
    dict_help = dict()

    for i in range(len(demo_prediction)):
        if demo_text_test[i][1] not in dict_help:
            dict_help[demo_text_test[i][1]] = [demo_prediction[i], i]
        else:
            if (dict_help[demo_text_test[i][1]][0] < demo_prediction[i]):
                dict_help[demo_text_test[i][1]] = [demo_prediction[i], i]
    answer_predicted = dict()
    answer_predicted[0] = (demo_text_test.T[0][dict_help[0][1]])
    return answer_predicted[0]

