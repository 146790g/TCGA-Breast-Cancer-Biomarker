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
import matplotlib.pyplot as plt
import pickle

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
from sklearn.model_selection import learning_curve
# Data Downloading



os.chdir('C:\Python Program\Python TCGA\BRCA_Python_Subtype\Raw Data')

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

y1=df.loc[:,'PAM50lite']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 
    
 
# Split Data into Training Data and Test Data

X=np.log(X+0.001)



# 6.1.2 パイプラインで変換器と推定器を結合する。

rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=5,
                          max_depth=2,
                          random_state=0,
                          n_jobs=1)


pipe_rf = Pipeline([('ms',MinMaxScaler()),
                    ('scl',StandardScaler()),
                    ('pca',PCA(n_components=2)),
                    ('rf',rf)])
    
    


train_size,train_scores,test_scores=learning_curve(estimator=pipe_rf,
                                                   X=X,
                                                   y=y,
                                                   train_sizes=np.linspace(0.1,1,10),
                                                   cv=10,
                                                   n_jobs=1)



# Training Data
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)

#Test Data
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)


plt.plot(train_size,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')

#信頼区間
plt.fill_between(train_size,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')

plt.plot(train_size,test_mean,color='red',marker='s',markersize=5,label='test accuracy')

#信頼区間
plt.fill_between(train_size,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')


plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7,0.99])
os.getcwd()

os.chdir('C:\Python Program\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('Learning_Curve.png')