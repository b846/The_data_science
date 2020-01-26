# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:12:42 2019

@author: admin
S33 L188
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

#lecture des données
data= pd.read_csv('1.01. Simple linear regression.csv')
data.describe()

# Création d'une régression
y = data['GPA']
x1 = data['SAT']
plt.scatter(x1,y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('DPA',fontsize=20)
plt.show()

#ajout de l'ordonnéeà l'origine
x= sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

# regression avec droite
plt.scatter(x1,y)
yhat= 0.0017*x1 +0.275
fig=plt.plot(x1, yhat, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('DPA',fontsize=20)
plt.show()