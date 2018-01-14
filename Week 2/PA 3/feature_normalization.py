#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def save_answers(some_list, decimals_count=2):
    for (n, answer) in enumerate(some_list):
        result_dir = os.path.join(os.path.dirname(__file__), 'answers')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fname = "q{}.txt".format(n+1)
        with open(os.path.join(result_dir, fname), 'w') as f:
            f.write(str(np.around(answer, decimals=decimals_count)))


def main():
    training_set = pd.read_csv("perceptron-train.csv", header=None)
    testing_set = pd.read_csv("perceptron-test.csv", header=None)


    _answers_columns = ['Y']
    _data_columns = ['X1', 'X2']
    _columns = _answers_columns + _data_columns
    training_set.columns = _columns
    testing_set.columns = _columns
    Y_test = testing_set[_answers_columns]
    Y_train = training_set[_answers_columns]
    X_test = testing_set[_data_columns]
    X_train = training_set[_data_columns]
    for x in [Y_test, Y_train, X_test, X_train]:
        print(x.head())

    clf = Perceptron(random_state=241)
    scaler = StandardScaler()

    clf.fit(X_train, Y_train)
    non_normalized_accuracy = accuracy_score(Y_test, clf.predict(X_test))
    print(non_normalized_accuracy)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf.fit(X_train_scaled, Y_train)
    normalized_accuracy = accuracy_score(Y_test, clf.predict(X_test_scaled))
    print(normalized_accuracy)
    print(normalized_accuracy-non_normalized_accuracy)
    answers = []
    answers.append(normalized_accuracy-non_normalized_accuracy)
    save_answers(answers, 3)


if __name__ == '__main__':
    main()
