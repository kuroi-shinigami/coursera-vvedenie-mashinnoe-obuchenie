#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC


def save_answers(some_list, decimals_count=2):
    for (n, answer) in enumerate(some_list):
        result_dir = os.path.join(os.path.dirname(__file__), 'answers')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fname = "q{}.txt".format(n+1)
        with open(os.path.join(result_dir, fname), 'w') as f:
            if type(answer) is not str:
                f.write(str(np.around(answer, decimals=decimals_count)))
            else:
                f.write(answer)


def main():
    data = pd.read_csv('svm-data.csv', header=None)
    print(data.head())
    _answers_columns = ['Y']
    _data_columns = ['X1', 'X2']
    _columns = _answers_columns + _data_columns
    data.columns = _columns
    Y = data[_answers_columns]
    X = data[_data_columns]

    clf = SVC(C=100000, random_state=241, kernel='linear')
    clf.fit(X, Y)

    a1 = ",".join([str(x+1) for x in clf.support_])  # Data science!
    print(clf.support_)
    print(a1)

    answers = [a1]
    save_answers(answers)


if __name__ == '__main__':
    main()
