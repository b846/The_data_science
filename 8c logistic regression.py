#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:01:47 2019

@author: b
lesson 236
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


# load the data
raw_data = pd.read_csv('2.01. Admittance.csv')
print(raw_data.head(6))
print(raw_data.describe(include='all'))
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data
y = data['Admitted']
x1 = data['SAT']

#plot of the data
plt.scatter(x1,y, color='C0')
plt.xlabel('SAT', fontsize=18)
plt.ylabel('Admitted',fontsize=18)
plt.show()

#plot with a regression line
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y, color='C0')
y_hat = x1*results_lin.params[1] + results_lin.params[0]

plt.plot(x1,y_hat,lw=2.5, color='C8')
plt.xlabel('SAT', fontsize=18)
plt.ylabel('Admitted',fontsize=18)
plt.show()


#plot a logistic regression
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

def f(x,b0,b1):
    return np.array(np.exp(b0+b1*x) / (1 + np.exp(b0+b1*x)))

f_sorted = np.sort(f(x1,results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y, color='C0')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted',fontsize=20)
plt.plot(x_sorted,f_sorted,color='C8')
plt.show()
