#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:50:00 2019

@author: b
type of ML
-logistic regression
-random forest
-neural network

ici, on va faire une regression linéaire
pb:we have standardized the dummies variables, it's a pb, 
because we lose the whole interpretability of dummies variables
we need to standardizing only the numerical data
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import numpy as np
import pandas as pd

#load the data np
 #raw_data_csv_np = np.loadtxt('Absenteeism-data.csv', delimiter = ',', dtype = 'str')
data_preprocessed_df = pd.read_csv('Absenteeism_preprocessed.csv')
print(data_preprocessed_df.head())
data_preprocessed_df['Month Value'][0]

#Create the targets
#np.where(condition, value if True, value if False): équivalent si
#using the median as a cut-off line is 'numerically stable and rigid'
targets = np.where(data_preprocessed_df['Absenteeism Time in Hours'] > 
                   data_preprocessed_df['Absenteeism Time in Hours'].median(), 1, 0)
data_preprocessed_df['Excessive Absenteeism'] = targets
print(data_preprocessed_df.head())

#A comment on the targets
#A balance of 45-55 is almost always sufficient for linear regressions
print(targets.sum() / targets.shape[0])
data_with_targets = data_preprocessed_df.drop(['Absenteeism Time in Hours', 'Date'], axis=1)
print(data_with_targets)

#Select the inputs for the regression
print(data_with_targets.shape)
#on va créer un df inputs (sans les targets)
unscaled_inputs = data_with_targets.iloc[:,:-1]

#Standardize the data
from sklearn.preprocessing import StandardScaler
absenteeism_scaler = StandardScaler()   #absenteeism_scaler is an empty StandardScaler
absenteeism_scaler.fit(unscaled_inputs) #scaling mecanism

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
print(scaled_inputs.shape)


#Split the data into train & test and shuffle
#sklearn.mode_selection.train_test_split(inputs, targets):split arrays or matrices into random train and test subsets
#random_state : la façon de mélanger les données reste la même
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state = 20)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#logistic regression with sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#Training the model
reg = LogisticRegression()      #regrestic logistic object
reg.fit(x_train, y_train)
print("the accuracy if the the model is: ", 100 * reg.score(x_train, y_train).round(3),"%")
#78% of the outputs match the targets

#Manually check the accuracy
model_outputs = reg.predict(x_train)
print(model_outputs)
print("the accuracy if the the model is: ", 100 * np.sum((model_outputs == y_train))/model_outputs.shape[0],"%")


#Finding the intercept an coefficients
print(reg.intercept_)
print(reg.coef_)
#type of unscaled_inputs : dataframe
#type of scaled_inputs: ndarray, because we use sklearn
print(unscaled_inputs.columns.values)
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
print(summary_table)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
print(summary_table)

#Interpreting the results
#l'équation est log(Odds) = b0 + b1*x1 +...
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
print(summary_table)
#DataFrame.sort_values(Series) : sort the values in a data frame with respect to a given column (Series)
summary_table.sort_values('Odds_ratio', ascending=False)
#the features with an odds_ratio close to 1 don't affect the outputs
