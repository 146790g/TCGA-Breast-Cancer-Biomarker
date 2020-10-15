# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:20:08 2017

@author: 146790

Ch 5.1.3  Logistic Regression using PCA transformation

"""

import numpy as np
from scipy import interp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
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


#射影行列をTrain Dataから構成し、それを用いて、Test DataをPCA変換する
X_test_pca=pca.transform(X_test_std)


# Model Fitting

rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=20,
                          max_depth=5,
                          random_state=0,
                          n_jobs=2)

rf = rf.fit(X_train_pca, y_train)

y_train_pred=rf.predict(X_train_pca)
y_test_pred=rf.predict(X_test_pca)


#決定領域をプロット)

from matplotlib.colors import ListedColormap

X=X_train_pca
y=y_train
classifier=rf
markers = ('s', 'x')
colors = ('red', 'blue')
cmap = ListedColormap(colors[:len(np.unique(y))])

subtype=('Basal','Non-Basal')

    # plot the decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

np.arange(1,100,0.5)
    
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))

xx1.shape
xx2.shape

np.array([xx1.ravel(), xx2.ravel()]).shape
np.array([xx1.ravel(), xx2.ravel()]).T.shape

Z = rf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)


fig = plt.figure(figsize=(8, 8))
# Create an axes instance
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


ax1.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)

ax1.set_xlim(xx1.min(), xx1.max())
ax1.set_ylim(xx2.min(), xx2.max())

    # plot class samples
for idx, cl in enumerate(np.unique(y)):
    ax1.scatter(x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.5, 
                color=cmap(idx),
                edgecolor='black',
                marker=markers[idx],
                label=subtype[idx])
        
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.legend(loc='lower left')
ax1.set_title('Fitting with Training Data')

     

# Plot for Test Data

X=X_test_pca
y=y_test
classifier=rf
markers = ('s', 'x')
colors = ('red', 'blue')
cmap = ListedColormap(colors[:len(np.unique(y))])
subtype=('Basal','Non-Basal')

    # plot the decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))

xx1.ravel().shape
    
Z = rf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)

ax2.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
ax2.set_xlim(xx1.min(), xx1.max())
ax2.set_ylim(xx2.min(), xx2.max())

    # plot class samples
for idx, cl in enumerate(np.unique(y)):
    ax2.scatter(x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.6, 
                color=cmap(idx),
                edgecolor='black',
                marker=markers[idx],
                label=subtype[idx])
        
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.legend(loc='lower left')
ax2.set_title('Fitting with Test Data')

plt.show()
    
os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('PCA & LR Fitting with Training&Test Data.png')
        