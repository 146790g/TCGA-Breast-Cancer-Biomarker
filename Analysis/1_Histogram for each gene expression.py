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
import numpy as np


#sklearn 
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data Downloading


os.chdir('D:\Python TCGA\LogisticRegression with PCA_Python')

df = pd.read_csv("LR_Analysis_Data.csv",index_col=0)
    
# Tumor Endothelial Marker

gene_x=list()
import csv
with open('Endothelial_Marker.csv', 'r') as f:
    c = csv.reader(f)
    for num,tau in c:
        gene_x.append(tau)
        
        
match={'BRCA':'Breast Cancer',
       'PAAD':'Pancreatic Adenocarcinoma', 
       'LUAD':'Lung Adenocarcinoma', 
       'LUSC':'Lung Squamous Cell Carcinoma',  
       'COAD':'Colon Adenocarcinoma', 
       'READ':'Rectum Adenocarcinoma Colorectal', 
       'STAD':'Stomach Adenocarcinoma', 
       'UCS':'Uterine Carcinosarcoma'}

TCGAtype=str('BRCA')

# Outlier Detection

import numpy as np
import matplotlib.pyplot as plt

NGSdata=TCGAtype+'_TPM.csv'
dat=pd.read_csv(NGSdata,index_col='gene')
sample=dat.shape[1]

k=0
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(len(gene_x)):
    j= i % 4 + 1
    ax = fig.add_subplot(2,2,j)
    ax.hist(df.ix[:,i],bins=100)    
    ax.set_xlabel(gene_x[i])
    ax.set_ylabel('Frequency')
           
    if (j ==4):
        k=k+1
        figname='Histogram for '+ TCGAtype + str(k) +'.png'
        fig.savefig(figname)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        








      
        
        
        
        
        
        
        
        
        
