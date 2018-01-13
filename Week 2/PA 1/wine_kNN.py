#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def classify_and_crossvalidate(X, Y, scaling=False):
    accuracies = []
    if scaling:
        X = scale(X)
    for k in range(1, 51):
        _score = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                                 X,
                                 Y,
                                 cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                 scoring='accuracy').mean()
        accuracies.append(_score)
        print(k, _score)
    optimal_accuracy = max(accuracies)
    optimal_k = accuracies.index(optimal_accuracy)+1
    print("Max accuracy: {} with k={}".format(optimal_accuracy, optimal_k))
    return optimal_k, optimal_accuracy


def save_answers(some_list):
    for (n, answer) in enumerate(some_list):
        result_dir = os.path.join(os.path.dirname(__file__), 'answers')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fname = "q{}.txt".format(n+1)
        with open(os.path.join(result_dir, fname), 'w') as f:
            f.write(str(np.around(answer, decimals=2)))


def main():
    data = pandas.read_csv('wine.data')
    # print(data.head())
    features = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash  ', 'Magnesium', 'Total phenols', 'Flavanoids',
                'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                'Proline']
    data.columns = ['Class'] + features
    Y = data['Class']
    X = data[features]
    # print(data.head())

    answers = []
    for x in classify_and_crossvalidate(X, Y, scaling=False):
        answers.append(x)
    for x in classify_and_crossvalidate(X, Y, scaling=True):
        answers.append(x)
    print(answers)
    save_answers(answers)


if __name__ == '__main__':
    main()
