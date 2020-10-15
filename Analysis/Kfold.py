# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:27:40 2019

@author: 146790
"""
from sklearn.model_selection import KFold
import pandas as pd
data = [('Ant'),('Beetle'),('Cat'),('Deer'),('Eagle'),('Fox')]
columns = ['name']
df = pd.DataFrame(data, columns=columns)

kf = KFold(n_splits=4)

for train, test in kf.split(df):
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    print('(train)', train_df)
    print('(test)', test_df)