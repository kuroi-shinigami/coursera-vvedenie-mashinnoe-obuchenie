#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


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
    newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
    X = newsgroups.data
    Y = newsgroups.target

    tf = TfidfVectorizer()
    X_tf = tf.fit_transform(X)

    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(random_state=241, kernel='linear')
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X_tf, Y)

    _min = None
    C = None
    for score, params in zip(gs.cv_results_['mean_test_score'], gs.cv_results_['params']):
        if _min is None:
            _min = score
            C = params
        if _min < score:
            _min = score
            C = params

    print("Found minumum score {} with {}".format(_min, C))

    clf = SVC(C=C['C'], random_state=241, kernel='linear')
    clf.fit(X_tf, Y)

    words = tf.get_feature_names()
    coef = pd.DataFrame(clf.coef_.data, clf.coef_.indices)
    top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
    a1 = [x for x in top_words]
    a1.sort()
    print(a1)
    a1 = ",".join(a1)
    answers = [a1]
    save_answers(answers)


if __name__ == '__main__':
    main()
