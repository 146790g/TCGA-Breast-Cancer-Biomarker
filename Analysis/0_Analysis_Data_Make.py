# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 16:04:12 2017

@author: 146790
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:48:43 2017

@author: 146790

BoxPlot for Subtype 
"""
import pandas as pd
import numpy as np
import pickle
import os
import xlrd
from scipy import stats
import csv
import numpy as np
import matplotlib.pyplot as plt


os.chdir('D:\Python TCGA\BRCA_Python_Subtype\Raw Data')
os.getcwd()


dat=pd.read_csv('TCGA_BRCA.csv',index_col=0).drop('Row.names',axis=1)

type(dat)

PAM50=dat.dropna(subset=['PAM50'])
TNBC=dat.dropna(subset=['TNBC'])
PAM50lite=dat.dropna(subset=['PAM50lite'])


# Pickle として出力

with open('PAM50.pickle',mode='wb') as f:
    pickle.dump(PAM50,f)
    
with open('TNBC.pickle',mode='wb') as f:
    pickle.dump(TNBC,f)
    
with open('PAM50lite.pickle',mode='wb') as f:
    pickle.dump(PAM50lite,f)
        