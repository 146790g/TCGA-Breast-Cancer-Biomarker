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

# summation of eigen values

tot=sum(eigen_vals)

#分散説明率

var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
len(var_exp)

#分散説明率の累積和を算出

cum_var_exp = np.cumsum(var_exp)

#分散説明率の棒グラフを作成

#1に始まり、１３に終わる。
plt.bar(range(1,14),var_exp[:13],alpha=0.5,align='center',label='individual explained variance')

#分散説明率の累積和の階段グラフを作成

plt.step(range(1,14),cum_var_exp[:13],where='mid',label='cumulative explained variance')

plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('Explained Variance Ratio_Curve.png')








