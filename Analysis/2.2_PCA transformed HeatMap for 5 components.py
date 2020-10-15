# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:20:08 2017

@author: 146790
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

# Data Downloading

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Raw Data')

with open('PAM50lite.pickle',mode='rb') as f:
    df=pickle.load(f)






# dfに列名を付ける

df2=df.drop(['PAM50','TNBC','PAM50lite'],axis=1)

#98%Quantile以上のデータには、98%Quantileの値を挿入する。

for i in range(len(df2.columns)-1):
    q=df2[df2.columns[i]].quantile(0.98)
    a=df2[df2.columns[i]]
    key=a>q
    #a[key] = q
    df2.loc[key,df2.columns[i]]=q


X=df2.values
type(X)

gene=df2.columns

y1=df.loc[:,'PAM50lite']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 



    
 
# Split Data into Training Data and Test Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0,random_state=1)

# Min Max Scaling

mms=MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)

 

 # Data Standardization
sc=StandardScaler()

X_train_std=sc.fit_transform(X_train_norm)
 
 
X_train_std.shape

# Covariance matrix
 
X2=X_train_std.T
cov_mat=np.cov(X_train_std.T)

# eigen value and eigen vectors

eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)

len(eigen_vals)
eigen_vecs.shape


eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
               


eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w=np.hstack((eigen_pairs[0][1][:,np.newaxis],
             eigen_pairs[1][1][:,np.newaxis],
             eigen_pairs[2][1][:,np.newaxis],
             eigen_pairs[3][1][:,np.newaxis],
             eigen_pairs[4][1][:,np.newaxis]))


a=eigen_pairs[4][1]

a.T

type(a)
a.shape
a.ndim

a2=eigen_pairs[4][1][:,np.newaxis]
a2.shape
a2.ndim

a2.T

# PCA transform

X_train_std.shape
w.shape

X_train_pca = X_train_std.dot(w)



    # 主成分の相関行列を計算する
correlation_matrix = np.corrcoef(X_train_pca.transpose())
    
correlation_matrix.shape
    
    
        # 主成分の相関行列をヒートマップで描く
feature_names = ['PCA{0}'.format(i)
                     for i in range(5)]

sns.heatmap(correlation_matrix, annot=True,
                xticklabels=feature_names,
                yticklabels=feature_names)

# another soluation

pca=PCA(n_components=5)
X_pca=pca.fit_transform(X) 
X_pca.shape

AR=df.loc[:,df.columns=='AR'].values
AR.shape

w=np.concatenate([X_pca,AR],axis=1)
w.shape

# 主成分の相関行列を計算する
correlation_matrix = np.corrcoef(w.transpose())
    
correlation_matrix.shape
    
    
 # 主成分の相関行列をヒートマップで描く
feature_names = ['PCA{0}'.format(i)
                     for i in range(5)]+['AR']

sns.heatmap(correlation_matrix, annot=True,
                xticklabels=feature_names,
                yticklabels=feature_names)




