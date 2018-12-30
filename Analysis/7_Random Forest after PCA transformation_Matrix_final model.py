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
#射影行列をTest Dataから構成し、それを用いて、Test DataをPCA変換する
X_test_pca=pca.transform(X_test_std)


# Model Fitting

rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=5,
                          max_depth=3,
                          random_state=0,
                          n_jobs=2)

rf = rf.fit(X_train_pca, y_train)

y_train_pred=rf.predict(X_train_pca)
y_test_pred=rf.predict(X_test_pca)

from sklearn.metrics import confusion_matrix

#Accuracy Score
as1=rf.score(X_train_pca, y_train)
as2=rf.score(X_test_pca, y_test)

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
fout=open('accuracy for final model.txt','w')
fout.writelines('Accuracy with Training Data: %.3f' % as1 + '\n')
fout.writelines('Acccuracy with Test Data: %.3f' % as2)
fout.close()



#混合行列 with Training Data
confmat=confusion_matrix(y_true=y_train,y_pred=y_train_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Evaluation with Training Data')
plt.tight_layout()

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('Evaluation Matrix with Training Data for final model.png')

#混合行列 with Test Data
confmat=confusion_matrix(y_true=y_test,y_pred=y_test_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Evaluation with Test Data')
plt.tight_layout()

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('Evaluation Matrix with Test Data for final model.png')


