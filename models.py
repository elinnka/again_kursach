import random
import pandas as pd
from metrics import *


class baseline():
    def F1_accuracy(self, data, answer_prepared):
        candidates_fromfile = [[] for i in range(5825)]
        for i in range(len(data)):
            candidates_fromfile[data['question_num'][i]].append(data['cand_ans'][i])

        predictedAnswer = []
        candidate_answer = candidates_fromfile
        matches = 0
        cont_num = 5825
        for numOfqas in range(cont_num):
            answer = ''
            if (len(candidate_answer[numOfqas]) == 0):
                answer = []
            else:
                predAnswer = random.randint(0, len(candidate_answer[numOfqas]) - 1)
                answer = candidate_answer[numOfqas][predAnswer]
            predictedAnswer.append(answer)
            if str(answer) == str(answer_prepared[numOfqas]):
                matches += 1

        pred = predictedAnswer[:cont_num]
        ans = answer_prepared[:cont_num]
        F1 = 0

        for i in range(cont_num):
            answer = set(str(pred[i]).split(" "))
            overlapped = len(answer.intersection(str(ans[i]).split(" ")))
            precision = overlapped / ((len(answer)))
            recall = overlapped / (len(str(ans[i]).split(" ")))
            if overlapped != 0:
                F1 += 2 * (precision * recall) / (precision + recall)

        # in %
        return F1 / cont_num * 100, matches / cont_num * 100


class Logreg():
    def __init__(self, data, indexes, answer_prepared):
        self.indexOfContext = indexes
        self.data = data
        self.answer_prepared = answer_prepared

    # фильтрация данных
    def filter(self, question_prep):
        data_new = []
        data = self.data
        for i in range(len(data)):
            if ((len(set(str(data['cand_ans'][i]).split(" ")) & set(
                    str(question_prep[data['question_num'][i]]).split(" "))) == 0)
                and (int(data['root_match'][i]) + data['tfidf_left'][i] + data['tfidf_right'][i] +
                     data['tfidf_span'][i] + data['tfidf_sentence'][i] +
                     data['tfidf_left_bigrams'][i] + data['tfidf_right_bigrams'][i] +
                     data['tfidf_span_bigrams'][i] + data['tfidf_sentence_bigrams'][i]) > 0 or data['is_answer'][i]) == 1:
                data_new.append(data.loc[i])
        data_help = pd.DataFrame(data_new).reset_index().drop(columns="index")
        return data_help

    # деление на выборки по контекстам
    def val_division_context_double(self, data):
        data = data.sample(frac=1).reset_index(drop=True)
        index = np.arange(self.indexOfContext[np.max(data['question_num'])])
        np.random.shuffle(index)
        train_size = int(np.round(0.8 * len(index)))
        val_size = train_size + int(np.round(0.1 * len(index)))
        data_train = []
        data_val = []
        data_test = []
        for i in range(len(data)):
            if self.indexOfContext[data['question_num'][i]] in index[:train_size]:
                data_train.append(data.loc[i])
            if self.indexOfContext[data['question_num'][i]] in index[train_size:val_size]:
                data_val.append(data.loc[i])
            if self.indexOfContext[data['question_num'][i]] in index[val_size:]:
                data_test.append(data.loc[i])
        data_test = pd.DataFrame(data_test).reset_index(drop=True)
        data_train = pd.DataFrame(data_train).reset_index(drop=True)
        data_val = pd.DataFrame(data_val).reset_index(drop=True)
        # data_train = filter(data)

        times = int(len(data_train) / np.sum(1 for i in range(len(data_train)) if data_train['is_answer'][i] == 1)) - 1
        data_train = data_train.loc[
            np.repeat(data_train.index.values, times * data_train['is_answer'] + 1)].reset_index(
            drop=True)
        data_train = data_train.sample(frac=1).reset_index(drop=True)

        x_test = np.array([list(map(int, data_test['root_match'])), data_test['length_span'], data_test['length_sent'],
                           data_test['length_left'], data_test['length_right'], data_test['tfidf_left'],
                           data_test['tfidf_right'], data_test['tfidf_span'], data_test['tfidf_sentence'],
                           data_test['span_word_freq'], data_test['tfidf_left_bigrams'],
                           data_test['tfidf_right_bigrams'],
                           data_test['tfidf_span_bigrams'], data_test['tfidf_sentence_bigrams']]).transpose()
        t_test = np.array([data_test['is_answer']]).transpose()
        x_text_test = np.array([data_test['cand_ans'], data_test['question_num']]).transpose()

        x_val = np.array([list(map(int, data_val['root_match'])), data_val['length_span'], data_val['length_sent'],
                          data_val['length_left'], data_val['length_right'], data_val['tfidf_left'],
                          data_val['tfidf_right'], data_val['tfidf_span'], data_val['tfidf_sentence'],
                          data_val['span_word_freq'], data_val['tfidf_left_bigrams'], data_val['tfidf_right_bigrams'],
                          data_val['tfidf_span_bigrams'], data_val['tfidf_sentence_bigrams']]).transpose()
        t_val = np.array([data_val['is_answer']]).transpose()
        x_text_val = np.array([data_val['cand_ans'], data_val['question_num']]).transpose()

        x_train = np.array(
            [list(map(int, data_train['root_match'])), data_train['length_span'], data_train['length_sent'],
             data_train['length_left'], data_train['length_right'], data_train['tfidf_left'],
             data_train['tfidf_right'], data_train['tfidf_span'], data_train['tfidf_sentence'],
             data_train['span_word_freq'], data_train['tfidf_left_bigrams'],
             data_train['tfidf_right_bigrams'],
             data_train['tfidf_span_bigrams'], data_train['tfidf_sentence_bigrams']]).transpose()
        x_text_train = np.array([data_train['cand_ans'], data_train['question_num']]).transpose()
        t_train = np.array([data_train['is_answer']]).transpose()

        x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train)
        x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test)
        x_val = (x_val - np.mean(x_val, axis=0)) / np.std(x_val)

        return x_train, t_train, x_text_train, x_val, t_val, x_text_val, x_test, t_test, x_text_test

    # деление на выборки по вопросам
    def val_division_double(self):
        data = self.data
        data = data.sample(frac=1).reset_index(drop=True)
        index = np.arange(np.max(data['question_num']))
        np.random.shuffle(index)
        train_size = int(np.round(0.8 * len(index)))
        val_size = train_size + int(np.round(0.1 * len(index)))
        data_train = []
        data_val = []
        data_test = []
        for i in range(len(data)):
            if data['question_num'][i] in index[:train_size]:
                data_train.append(data.loc[i])
            if data['question_num'][i] in index[train_size:val_size]:
                data_val.append(data.loc[i])
            if data['question_num'][i] in index[val_size:]:
                data_test.append(data.loc[i])
        data_test = pd.DataFrame(data_test).reset_index(drop=True)
        data_train = pd.DataFrame(data_train).reset_index(drop=True)
        data_val = pd.DataFrame(data_val).reset_index(drop=True)
        # data_train = filter(data)

        times = int(len(data_train) / np.sum(1 for i in range(len(data_train)) if data_train['is_answer'][i] == 1)) - 1
        data_train = data_train.loc[
            np.repeat(data_train.index.values, times * data_train['is_answer'] + 1)].reset_index(
            drop=True)
        data_train = data_train.sample(frac=1).reset_index(drop=True)

        x_test = np.array([list(map(int, data_test['root_match'])), data_test['length_span'], data_test['length_sent'],
                           data_test['length_left'], data_test['length_right'], data_test['tfidf_left'],
                           data_test['tfidf_right'], data_test['tfidf_span'], data_test['tfidf_sentence'],
                           data_test['span_word_freq'], data_test['tfidf_left_bigrams'],
                           data_test['tfidf_right_bigrams'],
                           data_test['tfidf_span_bigrams'], data_test['tfidf_sentence_bigrams']]).transpose()
        t_test = np.array([data_test['is_answer']]).transpose()
        x_text_test = np.array([data_test['cand_ans'], data_test['question_num']]).transpose()

        x_val = np.array([list(map(int, data_val['root_match'])), data_val['length_span'], data_val['length_sent'],
                          data_val['length_left'], data_val['length_right'], data_val['tfidf_left'],
                          data_val['tfidf_right'], data_val['tfidf_span'], data_val['tfidf_sentence'],
                          data_val['span_word_freq'], data_val['tfidf_left_bigrams'], data_val['tfidf_right_bigrams'],
                          data_val['tfidf_span_bigrams'], data_val['tfidf_sentence_bigrams']]).transpose()
        t_val = np.array([data_val['is_answer']]).transpose()
        x_text_val = np.array([data_val['cand_ans'], data_val['question_num']]).transpose()

        x_train = np.array(
            [list(map(int, data_train['root_match'])), data_train['length_span'], data_train['length_sent'],
             data_train['length_left'], data_train['length_right'], data_train['tfidf_left'],
             data_train['tfidf_right'], data_train['tfidf_span'], data_train['tfidf_sentence'],
             data_train['span_word_freq'], data_train['tfidf_left_bigrams'],
             data_train['tfidf_right_bigrams'],
             data_train['tfidf_span_bigrams'], data_train['tfidf_sentence_bigrams']]).transpose()
        x_text_train = np.array([data_train['cand_ans'], data_train['question_num']]).transpose()
        t_train = np.array([data_train['is_answer']]).transpose()

        x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train)
        x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test)
        x_val = (x_val - np.mean(x_val, axis=0)) / np.std(x_val)

        return x_train, t_train, x_text_train, x_val, t_val, x_text_val, x_test, t_test, x_text_test

    def sigm(self, b, x):
        return np.array(1 / (1 + np.exp(-x - b)))

    def lossFunc(self, t, y):
        res = 0
        for i in range(t.shape[0]):
            res += t[i] * np.log(y[i]) + (1 - t[i]) * np.log(1 - y[i])
        return -res / t.shape[0]

    def train(self, x, t, x_text_train, x_val, x_text_val, init_w=0.0001, gamma=0.0001, epochs=100):
        n, p = x.shape
        learning_rate = 0
        global losses, accuracy_plot
        losses = []
        accuracy_plot = []

        w = np.array([np.random.randn() * init_w for k in range(p)])
        b = np.random.randn() * init_w

        for i in range(epochs):
            if i % 10 == 0:
                print(i)
            help = self.sigm(b, x.dot(w)).reshape(1, len(t)) - t.reshape(1, len(t))
            dw = (help.dot(x) + learning_rate * w)[0]
            db = np.sum(help + learning_rate * b)
            w -= gamma * dw
            b -= gamma * db

            prediction_train = self.Predict(x, w, b)
            losses.append(self.lossFunc(t, prediction_train))

            accuracy_plot.append(F1_logreg(prediction_train, x_text_train, self.answer_prepared)[0])
            if i > 1 and np.abs(losses[i] - losses[i - 1]) <= 0.00001:
                print("STOP")
                break
            '''if i % 10 == 0:
                domain = np.arange(0, len(losses)) * 100
                plt.plot(domain, losses, label="val")
                plt.legend()
                plt.figure()
                plt.plot(domain, accuracy_plot, label="F1")
                plt.legend()
                plt.show()'''
            if i % 50 == 0:
                prediction = self.Predict(x_val, w, b)
                metrics = F1_logreg(prediction, x_text_val, self.answer_prepared)
        return w, b

    def Predict(self, x, w, b):
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y_pred[i] = self.sigm(b, w.T.dot(x[i]))
        return y_pred

    def evaluate(self):
        x_train, y_train, x_text_train, x_val, y_val, x_text_val, x_test, y_test, x_text_test = self.val_division_double()
        w, b = self.train(x_train, y_train, x_text_train, x_val, x_text_val)
        return F1_logreg(self.Predict(x_val, w, b), x_text_test,self.answer_prepared)
