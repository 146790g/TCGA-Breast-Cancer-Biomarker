# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:50:19 2017

@author: 146790
"""

# Ch6 ; モデル評価とハイパーパラメータのチューニングのベストプラクティス

import numpy as np
from scipy import interp
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#sklearn 
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Data Downloading

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Raw Data')

with open('PAM50lite.pickle',mode='rb') as f:
    df=pickle.load(f)

# dfに列名を付ける

df2=df.drop(['PAM50','TNBC','PAM50lite'],axis=1)

for i in range(len(df2.columns)-1):
    q=df2[df2.columns[i]].quantile(0.98)
    a=df2[df2.columns[i]]
    key=a>q
    #a[key] = q
    df2.loc[key,df2.columns[i]]=q


X=df2.values

X.shape


y1=df.loc[:,'PAM50lite']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 

 
# Split Data into Training Data and Test Data

X=np.log(X+0.001)

# 6.1.2 パイプラインで変換器と推定器を結合する。

rf=RandomForestClassifier(criterion='entropy',
                          max_depth=4,
                          n_estimators=5,
                          random_state=0,
                          n_jobs=1)


pipe_rf = Pipeline([('ms',MinMaxScaler()),
                    ('scl',StandardScaler()),
                    ('pca',PCA(n_components=2)),
                    ('rf',rf)])
    


pipe_rf.fit(X,y)

pipe_rf.steps[2][1].feature_importances_

importances=pipe_rf.feature_importances_

pipe_rf.named_steps['rf'].feature_importances_
pipe_rf.steps[3][1].get_feature_names


print(gs.best_estimator_)
print(gs.best_estimator_.rf__max_depth)
print(gs.best_score_)

print(gs.best_estimator_.get_params())


Predict = gs.predict(X)

gs.best_params_
gs.best_estimator_
gs.best_score_



