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
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

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




# 上の補正が適切に行われていることの検証のための
    fig,ax = plt.subplots(figsize=(10, 10))
    ax.hist(df2[df2.columns[2]],bins=100)
    plt.show()
    
 
# Split Data into Training Data and Test Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0,random_state=1)

# Min Max Scaling

mms=MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)

 

 # Data Standardization
 sc=StandardScaler()

 X_train_std=sc.fit_transform(X_train_norm)

# Covariance matrix
 
X2=X_train_std.T
cov_mat=np.cov(X_train_std.T)

# eigen value and eigen vectors

eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
               
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w=np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))

print('Projection Matrix W:\n',w)

# PCA transform

X_train_pca = X_train_std.dot(w)

X_train_pca = np.dot(X_train_std,w)

# plot for PCA transformed Data

colors=['r','b']
markers=['s','x']
subtype=["Basal","Non-Basal"]
      
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],
                c=c,label=subtype[l-1],marker=m)

plt.xlabel('PC 1',fontsize=15)
plt.ylabel('PC 2',fontsize=15)
plt.legend(loc='lower left',title="subtype")
plt.title('PCA Transformed Plot',fontsize=15)

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('PCA Transformed Plot.png')




