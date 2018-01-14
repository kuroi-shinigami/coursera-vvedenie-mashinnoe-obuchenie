#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale


def save_answers(some_list):
    for (n, answer) in enumerate(some_list):
        result_dir = os.path.join(os.path.dirname(__file__), 'answers')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fname = "q{}.txt".format(n+1)
        with open(os.path.join(result_dir, fname), 'w') as f:
            f.write(str(np.around(answer, decimals=2)))


def main():
    data, target = load_boston(return_X_y=True)
    X = scale(data)
    Y = target
    ps = []
    accuracies = []
    for p in np.linspace(1, 10, num=200):
        _score = cross_val_score(KNeighborsRegressor(n_neighbors=5, weights='distance', p=p),
                                 X,
                                 Y,
                                 cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                 scoring='neg_mean_squared_error').mean()
        ps.append(p)
        accuracies.append(_score)
        print(p, _score)

    optimal_acc = max(accuracies)
    ix = accuracies.index(optimal_acc)
    print("Max accuracy: {} with p={}".format(accuracies[ix], ps[ix]))

    answers = [ps[ix]]
    save_answers(answers)


if __name__ == '__main__':
    main()
