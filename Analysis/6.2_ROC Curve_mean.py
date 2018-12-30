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
    
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.8,random_state=1)
    
 
# Split Data into Training Data and Test Data

X=np.log(X+0.001)
X.shape

# 6.1.2 パイプラインで変換器と推定器を結合する。

pipe_lr = Pipeline([('ms',MinMaxScaler()),('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(penalty='l2',random_state=0))])


kfold = StratifiedKFold(y=y,n_folds=5,random_state=1)
len(kfold)


for k,(train,test) in enumerate(kfold):
    print(k,train,test)

len(train)
len(test)
len(y)
k

probas=pipe_lr.fit(X[train],y[train]).predict_proba(X[test])
probas.shape
fpr,tpr,thresholds = roc_curve(y[test],probas[:,1],pos_label=1)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area= %0.2f)' % (k+1,roc_auc))
    
thresholds

fig=plt.figure(figsize=(7,5))
mean_tpr= 0.0
mean_fpr=np.linspace(0,1,100)
all_tpr=[]


for k,(train,test) in enumerate(kfold):
    
    probas=pipe_lr.fit(X[train],y[train]).predict_proba(X[test])
    
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

plt.savefig('ROC_Curve.png')
