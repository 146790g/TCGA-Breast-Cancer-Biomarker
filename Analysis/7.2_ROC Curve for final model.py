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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import interp
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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

best_params={'pca__n_components': 5, 'rf__max_depth': 2, 'rf__n_estimators': 9}

rf=RandomForestClassifier(criterion='entropy',
                          n_estimators=best_params['rf__n_estimators'],
                          max_depth=best_params['rf__max_depth'],
                          random_state=0,
                          n_jobs=1)


pipe_rf = Pipeline([('ms',MinMaxScaler()),
                    ('scl',StandardScaler()),
                    ('pca',PCA(n_components=best_params['pca__n_components'])),
                    ('rf',rf)])

    


kfold = StratifiedKFold(n_splits=5,random_state=1)
kfold.split(X, y)

fig=plt.figure(figsize=(7,5))

mean_auc=[]

for k,(train,test) in enumerate(kfold.split(X, y)):
    print(k)
    probas=pipe_rf.fit(X[train],y[train]).predict_proba(X[test])
    
    fpr,tpr,thresholds = roc_curve(y[test],probas[:,1],pos_label=1)
    roc_auc=auc(fpr,tpr)
    mean_auc.append(roc_auc)
    plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area= %0.2f)' % (k+1,roc_auc))

    

mean_auc=np.mean(mean_auc)
print(mean_auc)

plt.grid()
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.title('Receiver Operrator Chracteristic')

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('ROC_Curve for final model.png')
