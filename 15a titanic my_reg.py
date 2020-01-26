#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:11:17 2019

@author: b
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
train_data_csv_df = pd.read_csv('titanic_train.csv')
pd.options.display.max_columns = None    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20
print(train_data_csv_df.head(6))
print(train_data_csv_df.describe(include='all'))

#Change the categorical data into numerical data
print(train_data_csv_df.columns.values)
print(train_data_csv_df['Embarked'].unique())
print(train_data_csv_df['Embarked'].value_counts())
train_data_csv_df['Embarked'] = train_data_csv_df['Embarked'].map({'S':0, 'C':1, 'Q':2})
print(train_data_csv_df['Sex'].unique())
print(train_data_csv_df['Sex'].value_counts())
train_data_csv_df['Sex'] = train_data_csv_df['Sex'].map({'male':0, 'female':1})


#Select the inputs for the regression
print(train_data_csv_df.shape)
print(train_data_csv_df.columns.values)
column_names = ['PassengerId', 'Survived', 'Ticket class', 'Name', 'Sex', 'Age', 'Parents',
       'Children', 'Ticket', 'Fare', 'Cabin', 'Embarked']
train_data_csv_df.columns = column_names
unscaled_data = train_data_csv_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parents'], axis = 1)

#Traitement des valeurs manquantes
#unscaled_data = unscaled_data.dropna(axis=0)            #on enlève toutes les lignes avec missing value
print(np.sum(np.isnan(unscaled_data)))
print(unscaled_data['Age'].mean())
unscaled_data['Age'].fillna((unscaled_data['Age'].mean()), inplace=True)
unscaled_data['Fare'].fillna((unscaled_data['Fare'].mean()), inplace=True)
unscaled_data['Embarked'].fillna(0, inplace=True)

#Create the targets
targets = unscaled_data['Survived']
unscaled_data = unscaled_data.drop(['Survived'], axis = 1)
print(targets.head(6))

#A comment on the targets
#A balance of 45-55 is almost always sufficient for linear regressions
print(targets.sum() / targets.shape[0])

####################################################################
#Standardize the numerical data
#from sklearn.preprocessing import StandardScaler
#absenteeism_scaler = StandardScaler()   #absenteeism_scaler is an empty StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):
    #this is the code for the StandardScaler with an additional argumets: the column to standardize
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y = None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

#choix des colonnes à standardiser
print(unscaled_data.columns.values)
columns_to_scale = ['Ticket class', 'Age', 'Children', 'Fare', 'Embarked']

#on standardise uniquement certaines colonnes
absenteeism_scaler = CustomScaler(columns_to_scale)     #on crée le mécanisme pour standardiser
absenteeism_scaler.fit(unscaled_data)                 #scaling mecanism

scaled_inputs = absenteeism_scaler.transform(unscaled_data)
print(scaled_inputs.shape)
print(scaled_inputs)
print(scaled_inputs.describe(include='all'))
x_train = unscaled_data
y_train = targets
###############################################################

#logistic regression with sklearn
from sklearn.linear_model import LogisticRegression
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
print(unscaled_data.columns.values)
feature_name = unscaled_data.columns.values
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
#DataFrame.sort_values(Series) : sort the values in a data frame with respect to a given column (Series)
summary_table = summary_table.sort_values('Odds_ratio', ascending=False)
#the features with an odds_ratio close to 1 don't affect the outputs
print(summary_table)

#####################################################
#on test le modèle
#load the data np
 #raw_data_csv_np = np.loadtxt('Absenteeism-data.csv', delimiter = ',', dtype = 'str')
test_data_csv_df = pd.read_csv('titanic_test.csv')
print(test_data_csv_df.head(6))
print(test_data_csv_df.describe(include='all'))
#on applique les mêmes transformations
print(test_data_csv_df.columns.values)
test_data_csv_df['Embarked'] = test_data_csv_df['Embarked'].map({'S':0, 'C':1, 'Q':2})
test_data_csv_df['Sex'] = test_data_csv_df['Sex'].map({'male':0, 'female':1})
print(test_data_csv_df.shape)
print(test_data_csv_df.columns.values)
column_names = ['PassengerId', 'Ticket class', 'Name', 'Sex', 'Age', 'Parents',
       'Children', 'Ticket', 'Fare', 'Cabin', 'Embarked']
test_data_csv_df.columns = column_names
unscaled_data = test_data_csv_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parents'], axis = 1)
unscaled_data['Age'].fillna((unscaled_data['Age'].mean()), inplace=True)
unscaled_data['Fare'].fillna((unscaled_data['Fare'].mean()), inplace=True)
unscaled_data['Embarked'].fillna(0, inplace=True)
print(np.sum(np.isnan(unscaled_data)))
#standardization
absenteeism_scaler.fit(unscaled_data)                 #scaling mecanism
scaled_inputs = absenteeism_scaler.transform(unscaled_data)
#Logistic regression
x_test = scaled_inputs
model_outputs = reg.predict(x_test)
df3 = test_data_csv_df['PassengerId']
raw_data = {'Survived': model_outputs}
df_a = pd.DataFrame(raw_data, columns = ['Survived'])

test_results = pd.concat([df3, df_a], axis=1)

#save our results
test_results.to_csv('Titanic_submissions.csv', index=False)









