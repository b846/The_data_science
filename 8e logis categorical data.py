#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:40:22 2019

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
raw_data = pd.read_csv('2.02. Binary predictors.csv')  
print(raw_data.head(6))
print(raw_data.describe(include='all'))
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data['Gender'] = data['Gender'].map({'Female':1, 'Male':0})
y = data['Admitted']
x1 = data[['SAT', 'Gender']]

#Regression
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()

#accuracy
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
results_log.predict()
np.array(data['Admitted'])
#if 80% of the predicted values coincide with the actual values, we say the model has 80% accuracy
results_log.pred_table()
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
print(cm_df)
#it's the confusion matrix 
cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
print(accuracy_train)

#TEST OF THE MODEL
test = pd.read_csv('2.03. Test dataset.csv')
print(test)
test['Admitted'] = test['Admitted'].map({'Yes':1, 'No':0})
test['Gender'] = test['Gender'].map({'Female':1, 'Male':0})
test_actual = test['Admitted']
test_data = test.drop(['Admitted'],axis=1)
test_data = sm.add_constant(test_data)
#test_data = test_data[x.columns.values]
print(test_data)

def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)           #predict the values    
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]       #summurize the values
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy

cm = confusion_matrix(test_data, test_actual, results_log)
print(cm)                 #cm and accurracy

cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
cm_df
