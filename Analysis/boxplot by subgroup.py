# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:32:04 2019

@author: 146790
"""

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
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

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




y1=df.loc[:,'PAM50lite']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 

sum(y1=='Basal')

X=df2.loc[y1=='Basal']

X.head()
type(X)


 
# Split Data into Training Data and Test Data
    

logX=np.log(X+0.0001)

mms=MinMaxScaler()
X_norm = mms.fit_transform(logX)

sc=StandardScaler()
X_norm_std=sc.fit_transform(X_norm)


# PCAのインスタンスを生成

pca=PCA(n_components=5)


# Training DataをPCA変換する

X_pca=pca.fit_transform(X_norm_std)

km = KMeans(n_clusters=2, 
            init='k-means++', 
            n_init=10, 
            max_iter=500,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X_pca)


X.columns

gene=X.columns
gene = sorted(gene)
k=0

# BoxPlot

gene[gene=='EGFR']


fig=plt.figure(figsize=(10,10))
fig.subplots_adjust(wspace=0.3,hspace=0.3) 

for i in range(len(gene)):
    
    M1=X.loc[y_km==0, X.columns==gene[i]]
    nM1=np.array(M1.values.flatten())
    M2=X.loc[y_km==1, X.columns==gene[i]]
    nM2=np.array(M2.values.flatten())
               
## combine these different collections into a list

    data_to_plot = [nM1,nM2]
               
    j =i % 6 +1
    mm='2'+'3'+str(j)
    ss=int(mm)
    
    ax = fig.add_subplot(ss)
    ax.boxplot(data_to_plot)
    ax.set_xticklabels(['SubClass1', 'SubClass2'])
    ax.set_title(gene[i])
    
    if j==6 and i>1:
        k=k+1
        figname='BoxPlot for TEM marker and Drug Resistance Transporter_'+str(k)+'.png'
        fig.savefig(figname, bbox_inches='tight')
        fig=plt.figure(figsize=(10,10))
        fig.subplots_adjust(wspace=0.3,hspace=0.3)
               