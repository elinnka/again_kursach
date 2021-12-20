import numpy as np


def F1_funct(ans, candidate):
    answer = set(str(candidate).split(" "))
    overlapped = len(answer.intersection(str(ans).split(" ")))
    precision = overlapped / len(answer)
    recall = overlapped / len(str(ans).split(" "))
    if overlapped != 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def F1_logreg(prediction, x_text_test, answer_prepared):
    dict_help = dict()

    for i in range(len(prediction)):
        if x_text_test[i][1] not in dict_help:
            dict_help[x_text_test[i][1]] = [prediction[i], i]
        else:
            if dict_help[x_text_test[i][1]][0] < prediction[i]:
                dict_help[x_text_test[i][1]] = [prediction[i], i]
    answer_predicted = dict()
    for key in dict_help.keys():
        answer_predicted[key] = (x_text_test.T[0][dict_help[key][1]])
    F1 = []
    accuracy = 0
    i = 0
    for key in answer_predicted.keys():
        F1.append(F1_funct(answer_prepared[key], answer_predicted[key]))
        if F1[i] == 1:
            accuracy += 1
        i += 1
    return np.sum(F1) / (len(F1)) * 100, accuracy / len(answer_predicted) * 100
