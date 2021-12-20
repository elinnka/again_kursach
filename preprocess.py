'''
Здесь извлечение веекторов характеристик из предобработанных данных
'''
import pandas as pd
import nltk
import numpy as np


def match_roots(root, quest_root):
    if len(root) == 0 or len(quest_root) == 0:
        return False
    return str(root) == str(quest_root)


def length_left_right(sentence, span):
    start = " ".join(sentence).find(str(span))
    word_in_sent = 0
    while start >= 0:
        word_in_sent += 1
        start -= len(sentence[word_in_sent - 1]) + 1
    end = word_in_sent + len(span.split(" ")) - 2
    return word_in_sent - 1, len(sentence) - end - 1


def tf_attempt(word, sentence):
    return ((sentence.count(' ' + word + ' ') + sentence.startswith(word + ' ') + sentence.endswith(
        ' ' + word)) / len(
        sentence.split(" ")))


def idf_attempt(context):
    dict_of_idf = dict()
    for sentence in context:
        for word in sentence.split(" "):
            dict_of_idf[word] = 0

    for sentence in context:
        for word in dict_of_idf.keys():
            if (sentence.count(' ' + word + ' ') > 0) | (sentence.startswith(word + ' ')) | (
                    sentence.endswith(' ' + word)):
                dict_of_idf[word] += 1

    for word in dict_of_idf.keys():
        if dict_of_idf[word] == 0:
            dict_of_idf[word] = 1
        dict_of_idf[word] = np.log(float(len(context)) / dict_of_idf[word])
    return dict_of_idf


def tfidf_attempt(word, sentence, context):
    dict_of_idf = idf_attempt(context)
    if word not in dict_of_idf:
        return 0
    return tf_attempt(word, sentence) * dict_of_idf[word]


def tfidf_sent_l_r_span(sentence, question, context, start, len):
    sum_left = 0
    sum_right = 0
    sum_span = 0

    sentence_words = sentence.split(" ")
    sentence_words_left = sentence_words[:start]
    sentence_words_right = sentence_words[start + len:]
    sentence_words_span = sentence_words[start:start + len]
    question_words = question.split(" ")

    for word in set(question_words) & set(sentence_words_left):
        sum_left += tfidf_attempt(word, sentence, context)
    for word in set(question_words) & set(sentence_words_right):
        sum_right += tfidf_attempt(word, sentence, context)
    for word in set(question_words) & set(sentence_words_span):
        sum_span += tfidf_attempt(word, sentence, context)
    sum_sent = sum_span + sum_left + sum_right

    return sum_sent, sum_left, sum_right, sum_span


def tfidf_span_notapp(span, sentence, question, context):
    sum_span = 0
    span_words = span.split(" ")
    for word in set(span_words):
        sum_span += tfidf_attempt(word, sentence, context)
    return sum_span


def tfidf_bigrams(sentence, question, context):
    bigrams_vocab_sentence = list(nltk.bigrams(sentence.split(" ")))
    bigrams_vocab_question = list(nltk.bigrams(question.split(" ")))
    bigrams_vocab_context = []
    for i in range(len(context)):
        bigrams_vocab_context.append(list(nltk.bigrams(context[i].split(" "))))

    dict_of_tfidf_bigrams = dict.fromkeys(set(bigrams_vocab_question) & set(bigrams_vocab_sentence), 0)

    for bigram in dict_of_tfidf_bigrams.keys():
        for sentence in bigrams_vocab_context:
            if bigram in sentence:
                dict_of_tfidf_bigrams[bigram] += 1
    for bigram in dict_of_tfidf_bigrams.keys():
        dict_of_tfidf_bigrams[bigram] = np.log(float(len(context)) / dict_of_tfidf_bigrams[bigram])
        dict_of_tfidf_bigrams[bigram] *= bigrams_vocab_sentence.count(bigram)
    return sum(dict_of_tfidf_bigrams.values())


def tfidf_bigrams_sent_l_r_span(sentence, question, context, start, len):
    sentence_words = sentence.split(" ")
    sentence_words_left = " ".join(sentence_words[:start])
    sentence_words_right = " ".join(sentence_words[start + len:])
    sentence_words_span = " ".join(sentence_words[start:start + len])

    sum_left = tfidf_bigrams(sentence_words_left, question, context)
    sum_right = tfidf_bigrams(sentence_words_right, question, context)
    sum_span = tfidf_bigrams(sentence_words_span, question, context)
    sum_sent = sum_span + sum_left + sum_right
    return sum_sent, sum_left, sum_right, sum_span


filename = "attempthelp_5825_1000.csv"
data_text = pd.read_csv(filename)

questions_prepared_help = data_text['question_prep']
sentencesInContext_for_quest_help = data_text['context_prep']
answer_prepared = data_text['answer_prep']
candidates_help = data_text['cand_ans']
questions = data_text['question']
contexts = data_text['context']
root_quest_prep_help = data_text['root_quest']
root_prep_help = data_text['root_prep']

num_of_prepared_contexts = 1000
num_of_prepared_questions = 5825

question_prep = []
candidates = []
root_prep = []
root_quest_prep = []
sentencesInContext_for_quest_prep = []

for i in range(num_of_prepared_questions):
    question_prep.append(
        questions_prepared_help[i].replace('[', '').replace('\'', '').replace(']', '').split(',')[0])
    sentencesInContext_for_quest_prep.append(
        sentencesInContext_for_quest_help[i].replace('[', '').replace('\'', '').replace(']', '').split(','))
    if len(candidates_help[i]) != 4:
        candidates.append(candidates_help[i].replace('[', '').replace("]]", '').split("], "))
    else:
        candidates.append([])
    root_prep.append(root_prep_help[i].replace('[', '').replace("]]", '').split("], "))
    root_quest_prep.append(root_quest_prep_help[i].replace('[', '').replace("]", ''))


filename = "/content/drive/MyDrive/train-v1.1.json"
data = pd.read_json(filename)


t_real = []
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

real_answer = []
feature_vector = []
cand_ans_df = []
num_quest = 0
num_sent_in_quest = 0
numbers_of_quest = []
numbers_of_sentence_in_quest = []
k = 0
for question in candidates:
    k += 1
    if num_quest % 100 == 0:
        print(num_quest)
    num_sent_in_quest = 0
    for sentence in question:
        for candidate in sentence.split(", "):
            l_r = length_left_right(sentencesInContext_for_quest_prep[num_quest][num_sent_in_quest].split(" "),
                                    candidate)
            length_left.append(l_r[0])
            length_right.append(l_r[1])
            length_span.append(len(candidate.split(" ")))
            length_sentence.append(len(sentencesInContext_for_quest_prep[num_quest][num_sent_in_quest].split(" ")))
            s_l_r_span = tfidf_sent_l_r_span(sentencesInContext_for_quest_prep[num_quest][num_sent_in_quest],
                                             question_prep[num_quest],
                                             sentencesInContext_for_quest_prep[num_quest],
                                             l_r[0],
                                             len(candidate.split(" ")))

            s_l_r_span_bigrams = tfidf_bigrams_sent_l_r_span(
                sentencesInContext_for_quest_prep[num_quest][num_sent_in_quest],
                question_prep[num_quest],
                sentencesInContext_for_quest_prep[num_quest],
                l_r[0],
                len(candidate.split(" ")))

            sum_tfidf_sentence.append(s_l_r_span[0])
            sum_tfidf_left.append(s_l_r_span[1])
            sum_tfidf_right.append(s_l_r_span[2])
            sum_tfidf_span.append(s_l_r_span[3])

            sum_tfidf_sentence_bigrams.append(s_l_r_span_bigrams[0])
            sum_tfidf_left_bigrams.append(s_l_r_span_bigrams[1])
            sum_tfidf_right_bigrams.append(s_l_r_span_bigrams[2])
            sum_tfidf_span_bigrams.append(s_l_r_span_bigrams[3])

            sum_tfidf_span_notapp.append(tfidf_span_notapp(str(candidate),
                                                           sentencesInContext_for_quest_prep[num_quest][
                                                               num_sent_in_quest],
                                                           question_prep[num_quest],
                                                           sentencesInContext_for_quest_prep[num_quest]))
            if (num_sent_in_quest < len(root_prep[num_quest])):
                root_match.append(int(match_roots(root_prep[num_quest][num_sent_in_quest], root_quest_prep[num_quest])))
            else:
                root_match.append(0)
            real_answer.append(answer_prepared[num_quest])
            cand_ans_df.append(candidate)
            numbers_of_sentence_in_quest.append(num_sent_in_quest)
            numbers_of_quest.append(num_quest)
            t_real.append(str(answer_prepared[num_quest]) == str(candidate))

        num_sent_in_quest += 1
    num_quest += 1

lengths = pd.DataFrame({'question_num': numbers_of_quest,
                        'sentence_in_context_num': numbers_of_sentence_in_quest,
                        'is_answer': t_real,
                        'cand_ans': cand_ans_df,
                        'real_ans': real_answer,
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
data = lengths

# запись таблицы в файл
table_to_download = data

table_to_download.to_csv('table_final3_754115_5825.csv')
with open ('table_final3_754115_5825.csv', 'w') as f:
    f.write(table_to_download)
