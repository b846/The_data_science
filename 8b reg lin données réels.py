#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:40:20 2019

@author: b
lesson 227
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
sns.set()

# load the data
raw_data = pd.read_csv('1.04. Real-life example.csv')
print(raw_data.head(6))
print(raw_data.describe(include='all'))

#PREPROCESSING
#determining the data of interest
data = raw_data.drop(['Model'],axis=1)          #on enlève le model, car la fréquence est trop faible
data.describe(include='all')

#dealing with missing values
#s'il manque plus de5% des données, alors, on peut enlever la colonne
print(data.isnull().sum())
data_no_mv = data.dropna(axis=0)            #on enlève toutes les lignes avec missing value
data_no_mv.describe(include='all')

#exploring the PDFs
sns.distplot(data_no_mv['Price'])            #exponential distribution

#on enlève les valeurs extrêmes
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')

#on enlève les valeurs extrêmes
q = data_1['Year'].quantile(0.01)
data_2 = data_1[data_no_mv['Price']>q]
data_2.describe(include='all')

#exploring the PDFs
sns.distplot(data_2['Year'])            #exponential distribution

#on ré-index
data_cleaned = data_2.reset_index(drop=True)
data_cleaned.describe(include='all')


### Checking the OLS assumptions
#1- Linéarité
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()
sns.distplot(data_cleaned['Price'])            #exponential distribution
#on voit que les données semblent exponentielles
#on va linéariser les données avec np.log
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

#plot of the data
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('log_price and year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('log_price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('log_price and Mileage')
plt.show()              #les données semblent linéaires

data_cleaned = data_cleaned.drop(['Price'], axis=1)
#2- no endogeneity, signifie que l'erreur et la variable sont corrélées
#3- normality and homoscedasticity
#  -normality
#  - zero mean
#  -homoscedasticity (deux droites peuvent être traçées pour contenir les données), valable avec la log transformation
#4 no autocorrelation, the observations that we have are not coming from time series data or panel data
#5- Multicollinearity
data_cleaned.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]      #we define the features we want to check from multicollinearity
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns

data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
#VIF varie entre 1 et infini
# VIF = 1 : pas de multicollinearity
# VIF entre 1 et 5 : perfect okay
# VIF > 5 : unacceptable: variable à enlever

#229 create dummy variables, variable muette
#pd.get_dummies(df, [,drop_first]) spot all categorical variables and create dummunies automatically
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
data_with_dummies.head()

#rearrangement des colonnes
#pour réarranger les colonnes, on change l'ordre des titres
data_with_dummies.columns.values
cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
data_preprocessed

#230 linear regression model
#declare the inputs and the targets
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

#Scale the data
#it's not usually recommended to standardize dummy variables
#scaling has no effect on the predictive power of dummies,
#once scaled, though, they lose all their dummy meaning
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

#create the regression between the targets and the regression
#if the target is 7, we want the prediction to be 7
#if the target is 10, we want the prediction to be 10
#we want to draw a 45° line between targets and prediction
#the closer our scatter plot to this line, the better the model
reg = LinearRegression()         # création d'une classe  linearRegression
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)         # graphe y_hat en fct de y_train
plt.xlabel('Targets (y_train)', fontsize=18)
plt.ylabel('Predictions (y_hat)',fontsize=18)
plt.xlim(6.13)
plt.ylim(6.13)
plt.show()

#on affiche la différence entre y_train et y_hat
#l'erreur doit suivre une distribution normale d'espérance nulle
sns.distplot(y_train - y_hat)
plt.title("Residual PDF", size=18)

reg.score(x_train, y_train)
#our result is explaining 75%68 of the variability of the data

#création des weights
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary ['weights'] = reg.coef_
print(reg_summary.head(10))
#continuous variables, a positive weight shows that as a feature increases is value, so do the log_price and 'price' respectively
#continus variables, a negative weight shows that as a feature increases is value, log_price and 'price' decrease

print(data_cleaned['Brand'].unique())
#dummy variables, a positive weight sows that the respective category (Brand) is more expensive than the benchmark (Audi)
#dummy variables, a negative weight sows that the respective category (Brand) is less expensive than the benchmark (Audi)
#dummy are only compared to their respective benchmark

#What are the reference (benchmark) categories for each categorical variable?
#data_cleaned['name_of_categorical_variable'].unique() and find which ones are missing.

#The benchmark categories are as follows:
#Brand -> 'Audi'
#Body -> 'crossover'
#Engine Type -> 'Diesel'
#Registration -> 'no'
#Naturally, if you reorder the categorical variables prior to using .get_dummies() you will be able to customize the reference category.

#TESTING
#les valeurs proche de la droite d:y=x sont bien estimées, les autres non
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)         # graphe y_hat en fct de y_train, alpha est lopacité
plt.xlabel('Targets (y_train)', fontsize=18)
plt.ylabel('Predictions (y_hat)',fontsize=18)
plt.xlim(6.13)
plt.ylim(6.13)
plt.show()

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])               #performance
#we know have the predictions expressed in price
df_pf['Target'] = np.exp(y_test)                        #lot of missing values
print(y_test)
y_test = y_test.reset_index(drop=True)
df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Difference%'] = np.absolute(df_pf['Predictions']/df_pf['Target']*100)

df_pf.describe()
#the minimum of difference in % is 3%, so the output was spot on
#the maximum difference in % is very high
df_pf.sort_values(by=['Difference%'])
pd.options.display.max_rows = 15            #affichage de 15 lignes
pd.set_option('display.float_format', lambda x:'%.2f' % x)
df_pf.sort_values(by=['Difference%'])


#how to improve the model
#1. use a different set of variables
#2. remove bigger part of the outliers
#3. use different kinds of transformation