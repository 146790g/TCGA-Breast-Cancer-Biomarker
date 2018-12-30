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
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score


# Fold Specification
os.chdir('D:\Python TCGA\LogisticRegression with PCA_Python')

# Tumor Endothelial Marker
gene_x=list()
import csv
with open('Endothelial_Marker.csv', 'r') as f:
    c = csv.reader(f)
    for num,tau in c:
        gene_x.append(tau)




# dfに列名を付ける
        
df = pd.read_csv("LR_Analysis_Data.csv",index_col=0)

for i in range(len(gene_x)-1):
    q=df[gene_x[i]].quantile(0.95)
    a=df[gene_x[i]]
    key=a>q
    #a[key] = q
    df.ix[key,gene_x[i]]=q

df.describe()

df2=df.drop('class',axis=1)
X=df2.values

y1=df.loc[:,'class']

le=LabelEncoder()
y=le.fit_transform(y1)

# 上の補正が適切に行われていることの検証のための
    #fig,ax = plt.subplots(figsize=(10, 10))
    #ax.hist(df[gene_x[2]],bins=100)
    #plt.show()
    
 
# Split Data into Training Data and Test Data
    

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.8,random_state=1)

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

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Model Fitting

lr=LogisticRegression(penalty='l2',random_state=0)
lr = lr.fit(X_train_pca, y_train)

y_train_pred=lr.predict(X_train_pca)
y_test_pred=lr.predict(X_test_pca)




#決定領域をプロット)

from matplotlib.colors import ListedColormap

X=X_train_pca
y=y_train
classifier=lr
markers = ('s', 'x')
colors = ('red', 'blue')
cmap = ListedColormap(colors[:len(np.unique(y))])
subtype=('0-25','75-100')

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=subtype[idx])
        
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.title('Fitting with Training Data')


plt.savefig('PCA & LR Fitting with Training Data.png')
        

# Plot for Test Data

X=X_test_pca
y=y_test
classifier=lr
markers = ('s', 'x')
colors = ('red', 'blue')
cmap = ListedColormap(colors[:len(np.unique(y))])
subtype=('0-25','75-100')

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=subtype[idx])
        
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.title('Fitting with Test Data')

plt.savefig('PCA & LR Fitting with Test Data.png')
        