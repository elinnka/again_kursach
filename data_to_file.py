'''
Здесь предобработка данных - удаление пунктуации, фильтрация, токенизация
'''
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
import benepar
from benepar.spacy_plugin import BeneparComponent
import spacy
import nltk

nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords


def preprocess(data):

    benepar.download('benepar_en3')

    nlp = spacy.load('en')
    nlp.add_pipe(BeneparComponent('benepar_en3'))

    questions_start = []
    answers_start = []
    context_start = []
    answer_start = []

    for i in range(data['data'].shape[0]):
        for j in data['data'][i]['paragraphs']:
            for qas in j['qas']:
                questions_start.append(qas['question'])
                context_start.append(j['context'])
                if len(qas['answers']) > 0:
                    answers_start.append(qas['answers'][0]['text'])
                    answer_start.append(qas['answers'][0]['answer_start'])
                else:
                    answers_start.append(qas['plausible_answers'][0]['text'])
                    answer_start.append(qas['plausible_answers'][0]['answer_start'])

    df = pd.DataFrame(
        {"context": context_start, "question": questions_start, "answer_start": answer_start,
         "answer_text": answers_start})

    # делим на контексты
    sentences = list(df["context"].drop_duplicates().reset_index(drop=True))

    # индексы контекстов - чтобы узнать какой контекст используется в определеннм вопросе
    index_of_context = []
    for i in range(len(df)):
        index_of_context.append(sentences.index(df['context'][i]))

    # фильтрация слов
    sentences_filtred = []
    for sentence in sentences:
        sentences_filtred.append(
            ' '.join(
                map(str, [word for word in sentence.split(" ") if word not in stopwords.words('english')])).lower())

    # деление на предложения
    sentences_in_context_div = []
    for i in range(len(sentences_filtred)):
        blob = TextBlob(" ".join(sentences_filtred[i:i + 1]))
        sentences_in_context_div .append([item.raw for item in blob.sentences])

    # убираем пунктуацию
    sent_wtht_punct_inContext = []
    for i in range(len(sentences_in_context_div)):
        sentence_wtht_punct = []
        for j in range(len(sentences_in_context_div[i])):
            sentence_wtht_punct.append(tokenizer.tokenize(sentences_in_context_div[i][j][:-1]))
        sent_wtht_punct_inContext.append(sentence_wtht_punct)

    # делаем строки
    sentencesInContext = []
    for sentence in sent_wtht_punct_inContext:
        sentences = []
        for sent in sentence:
            sentences.append(' '.join(map(str, sent)))
        sentencesInContext.append(sentences)

    # обрабатываем вопросы и делаем из них как контексты
    questions_filtred = []
    for question in list(df['question']):
        questions_filtred.append(
            ' '.join([word.lower() for word in question.split(" ") if word not in stopwords.words('english')]))

    questions_in_context_div = []
    for i in range(len(questions_filtred)):
        blob = TextBlob(" ".join(questions_filtred[i:i + 1]))
        questions_in_context_div.append(str([item.raw for item in blob.sentences]))

    questions_wtht_punct = []
    for i in range(len(questions_in_context_div)):
        questions_wtht_punct.append(tokenizer.tokenize(questions_in_context_div[i][:-1]))

    questions_prepared = []
    for el in questions_wtht_punct:
        questions_prepared.append(' '.join(el).lower())

    # обрабатываем ответы
    answer_filtred = []
    for answer in list(df["answer_text"]):
        answer_filtred.append(
            ' '.join([word.lower() for word in answer.split(" ") if word not in stopwords.words('english')]))

    answer_in_context_div = []
    for i in range(len(answer_filtred)):
        blob = TextBlob(" ".join(answer_filtred[i:i + 1]))
        answer_in_context_div.append(str([item.raw for item in blob.sentences]))

    answer_wtht_punct = []
    for i in range(len(answer_in_context_div)):
        answer_wtht_punct.append(tokenizer.tokenize(answer_in_context_div[i][:-1]))

    answer_prepared = []
    for el in answer_wtht_punct:
        answer_prepared.append(' '.join(el).lower())

    num_of_prepared_contexts = 1000
    sents = []
    for i in range(num_of_prepared_contexts):
        context = sentencesInContext[i]
        sents_in_sentence = []
        for sentence in context:
            sents_morethan1 = []
            if sentence != "":
                for el in list(nlp(sentence).sents):
                    sents_morethan1.extend(list(el._.constituents))
            sents_in_sentence.append(sents_morethan1)
        if len(sents_in_sentence) == 0:
            sents_in_sentence = [""]
        sents.append(sents_in_sentence)


    root_sentence_in_context = []
    for i in range(num_of_prepared_contexts):
        if (i % 10 == 0):
            print(i)
        root_sentence_in_context_help = []
        sentence = sentences[i].replace("?", ".").replace(". ", ".").split(".")
        for j in range(len(sentence)):
            doc_sentences_in_context = nlp(sentence[j])
            root_sentence = []
            for el in doc_sentences_in_context.sents:
                root_sentence.append(el.root)
            root_sentence_in_context_help.append(root_sentence)
        root_sentence_in_context.append(root_sentence_in_context_help)

    # это потому что нам без разницы до предложения в контексте
    num_of_prepared_questions = index_of_context.index(num_of_prepared_contexts)
    sentences_in_context_prep = [sentencesInContext[index_of_context[i]] for i in range(num_of_prepared_questions)]

    candidates = []
    root_prep = [root_sentence_in_context[index_of_context[i]] for i in range(num_of_prepared_questions)]

    for i in range(num_of_prepared_questions):
        ca_sentence = []
        for j in range(len(sentences_in_context_prep[i])):
            ca_sentence.append(sents[index_of_context[i]][j])
        if len(ca_sentence) == 0:
            ca_sentence = [""]
        candidates.append(ca_sentence)
    candidates_main = candidates

    # это потому что нам без разницы до предложения в контексте
    num_of_prepared_questions = index_of_context.index(num_of_prepared_contexts)
    sentencesInContext_for_quest = [sentencesInContext[index_of_context[i]] for i in range(num_of_prepared_questions)]

    candidates = []
    sentences_in_context_prep = []
    root_prep = []

    for i in range(num_of_prepared_questions):
        if i % 100 == 0:
            print(i)
        context = sentencesInContext_for_quest[i]
        ca_sentence = []
        sentencesInContext_prep_help = []
        root_prep_help = []
        for j in range(len(context)):
            ca_sentence.append(sents[index_of_context[i]][j])
            sentencesInContext_prep_help.append(context[j])
            doc_sentencesInContext = nlp(df['context'][i].replace("?", ".").replace(". ", ".").split(".")[j])
            root_sentence = []
            for el in doc_sentencesInContext.sents:
                root_sentence.append(el.root)
            root_prep_help.append(root_sentence)
        if len(ca_sentence) == 0:
            ca_sentence = [""]
        sentences_in_context_prep.append(sentencesInContext_prep_help)
        candidates.append(ca_sentence)
        root_prep.append(root_prep_help)

    # корни деревьев в вопросах!
    root_quest_prep = []
    for i in range(num_of_prepared_questions):
        if (i % 100 == 0):
            print(i)
        doc_sentencesInContext = nlp(df['question'][i])
        root_sentence = []
        for el in doc_sentencesInContext.sents:
            root_sentence.append(el.root)
        root_quest_prep.append(root_sentence)


    file_to_download = pd.DataFrame({'question': df['question'][:num_of_prepared_questions],
                                     'context': df['context'][:num_of_prepared_questions],
                                     'answer': df['answer_text'][:num_of_prepared_questions],
                                     'question_prep': questions_prepared[:num_of_prepared_questions],
                                     'context_prep': sentences_in_context_prep[:num_of_prepared_questions],
                                     'answer_prep': answer_prepared[:num_of_prepared_questions],
                                     'cand_ans': candidates[:num_of_prepared_questions],
                                     'root_prep': root_prep[:num_of_prepared_questions],
                                     'root_quest': root_quest_prep[:num_of_prepared_questions]})

    file_to_download.to_csv('attempthelp_5825_1000.csv')