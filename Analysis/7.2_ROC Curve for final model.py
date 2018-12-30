# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 08:51:50 2017

@author: 146790
"""

# 6.5.3 ROC 曲線をプロットする（page. 186）

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
from sklearn.learning_curve import learning_curve

# Data Downloading

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Raw Data')

with open('PAM50lite.pickle',mode='rb') as f:
    df=pickle.load(f)



# dfに列名を付ける

df2=df.drop(['PAM50','TNBC','PAM50lite'],axis=1).dropna(how='all',axis=1)
df2.shape
df.shape

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
                          max_depth=3,
                          random_state=0,
                          n_jobs=1)


pipe_rf = Pipeline([('ms',MinMaxScaler()),
                    ('scl',StandardScaler()),
                    ('pca',PCA(n_components=2)),
                    ('rf',rf)])
    

kfold = StratifiedKFold(y=y,n_folds=5,random_state=1)
len(kfold)

fig=plt.figure(figsize=(7,5))


for k,(train,test) in enumerate(kfold):
    
    probas=pipe_rf.fit(X[train],y[train]).predict_proba(X[test])
    
    fpr,tpr,thresholds = roc_curve(y[test],probas[:,1],pos_label=1)
    
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area= %0.2f)' % (k+1,roc_auc))
    
    
plt.grid()
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.title('Receiver Operrator Chracteristic')

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('ROC_Curve for final model.png')
