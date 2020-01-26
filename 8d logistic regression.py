#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:04:29 2019

@author: b
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
import sklearn
from sklearn.linear_model import LinearRegression


#Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq, df)

# load the data
raw_data = pd.read_csv('2.01. Admittance.csv')  
raw_data.head(6)
raw_data.describe(include='all')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
y = data['Admitted']
x1 = data['SAT']

#Regression
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

#summary
results_log.summary()
