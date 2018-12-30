# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:50:09 2018

@author: 146790
"""


# Ch6 ; モデル評価とハイパーパラメータのチューニングのベストプラクティス

import numpy as np
from scipy import interp
import os
import pandas as pd
import matplotlib.pyplot as plt

#sklearn 
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.learning_curve import validation_curve

# Data Downloading

os.chdir('D:\Python TCGA\Python PANCAN')

# load
with open('data.pickle', mode='rb') as f:
    df=pickle.load(f)

gene_x=df.columns[:df.shape[1]-1]


df2=df.drop('Class',axis=1)
X=df2.values

y1=df.loc[:,'Class']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 

X.shape


 
# Split Data into Training Data and Test Data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.8,random_state=1)

X=np.log(X+0.001)

# 6.1.2 パイプラインで変換器と推定器を結合する。

rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=10,
                          max_depth=3,
                          random_state=0,
                          n_jobs=1)


pipe_rf = Pipeline([('ms',MinMaxScaler()),
                    ('scl',StandardScaler()),
                    ('rf',rf)])

pipe_rf = pipe_rf.fit(X_train, y_train)

y_train_pred=pipe_rf.predict(X_train)
y_test_pred=pipe_rf.predict(X_test)

#Accuracy Score
as1=pipe_rf.score(X_train, y_train)
as2=pipe_rf.score(X_test, y_test)

print('Accuracy with Training Data: %.3f' % as1)
print('Acccuracy with Test Data: %.3f' % as2)


#混合行列 with Training Data
confmat=confusion_matrix(y_true=y_train,y_pred=y_train_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Evaluation with Training Data')
plt.tight_layout()
plt.savefig('Evaluation Matrix with Training Data.png')

#混合行列 with Test Data
confmat=confusion_matrix(y_true=y_test,y_pred=y_test_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Evaluation with Test Data')
plt.tight_layout()
plt.savefig('Evaluation Matrix with Test Data.png')



