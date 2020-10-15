# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:38:24 2019

@author: 146790
"""

import numpy as np
from scipy import interp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
#sklearn 
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

y1=df.loc[:,'PAM50lite']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 

 
# Split Data into Training Data and Test Data
    

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train.shape)
print(X_test.shape)

logX_train=np.log(X_train+0.0001)
logX_test=np.log(X_test+0.0001)

mms=MinMaxScaler()
X_train_norm = mms.fit_transform(logX_train)
X_test_norm = mms.fit_transform(logX_test)

sc=StandardScaler()
X_train_std=sc.fit_transform(X_train_norm)
X_test_std=sc.transform(X_test_norm)

# PCAのインスタンスを生成

pca=PCA(n_components=2)

# Training DataをPCA変換する

X_train_pca=pca.fit_transform(X_train_std)
#射影行列をTest Dataから構成し、それを用いて、Test DataをPCA変換する
X_test_pca=pca.transform(X_test_std)


# Model Fitting and Accuracy for Training Data

param_range=[1,2,3,5,7,10,15,20]
as1=[]

for n_estimators in param_range:
    rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=n_estimators,
                          max_depth=2,
                          random_state=0,
                          n_jobs=2)
    rf.fit(X_train_pca, y_train)
    as1.append(rf.score(X_train_pca, y_train))

    

# Model Fitting and Accuracy for Test Data
   
param_range=[1,2,3,5,7,10,15,20]
as2=[]

for n_estimators in param_range:
    rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=n_estimators,
                          max_depth=2,
                          random_state=0,
                          n_jobs=2)
    rf.fit(X_test_pca, y_test)
    as2.append(rf.score(X_test_pca, y_test))


#plt.show()

# 

plt.plot(param_range,as1,label="accuracy for training data")
plt.plot(param_range,as2,label="accuracy for test data")

plt.grid()
plt.title("Simgple Tuning for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.legend()
#plt.show()

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('Simgple Tuning for n_estimators.png')

