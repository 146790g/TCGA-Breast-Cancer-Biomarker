+# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:34:31 2019

@author: 146790
"""

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
from sklearn.tree import DecisionTreeClassifier


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

X.shape
len(y)

y1=df.loc[:,'PAM50lite']

le=LabelEncoder()
y=le.fit_transform(y1)
le.classes_
le.transform(le.classes_) 

 
# Split Data into Training Data and Test Data

X=np.log(X+0.001)

# 6.1.2 パイプラインで変換器と推定器を結合する。



from sklearn import tree

clf=tree.DecisionTreeClassifier(criterion='entropy')



param_range=[1,2,3,4,5,6,7,8,9,10,]
    
train_scores,test_scores=validation_curve(estimator=clf,
                                                   X=X,
                                                   y=y,
                                                   param_name='max_depth',
                                                   param_range=param_range,
                                                   cv=10)

# Training Data
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)

#Test Data
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')

#信頼区間
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')

plt.plot(param_range,test_mean,color='red',marker='s',markersize=5,label='test accuracy')

#信頼区間
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')


plt.grid()
plt.title('Decision Tree')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.6,1.02])
os.getcwd()

os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Analysis')
plt.savefig('DecisionTree_Validation_Curve_max_depth.png')

