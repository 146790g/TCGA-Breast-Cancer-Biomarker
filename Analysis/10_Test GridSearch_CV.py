# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:22:38 2019

@author: 146790
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=['target'])
train_x, test_x, train_y, test_y = train_test_split(X, y)


# グリッドサーチ(パラメータ候補指定)用のパラメータ10種
paramG = {'n_estimators':[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],'max_depth':[2,4,6,8,10]}


# モデル生成。上から順に、通常のランダムフォレスト、グリッドサーチ・ランダムフォレスト、
# ランダムサーチ・ランダムフォレスト。

gs = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                  param_grid=paramG,scoring='r2', cv=3)


# 各モデルに学習を行わせる。
gs.fit (train_x, train_y.as_matrix().ravel())

gs.best_estimator_.n_estimators
gs.best_estimator_.max_depth

r2_score(test_y, gs.predict(test_x))



