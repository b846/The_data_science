#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:14:43 2019

@author: b
we will perform a decision tree
Decision trees compute entropy in the information system. If you peform a decision tree 
on dataset, the variable importances_ contains important information on what columns of 
data has large variances thus contributing to the decision.
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#load the data np
 #raw_data_csv_np = np.loadtxt('Absenteeism-data.csv', delimiter = ',', dtype = 'str')
df = pd.read_csv('titanic_train.csv')
pd.options.display.max_columns = None    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20
df.head(6)
df.describe(include='all')
column_names = df.columns
print(df.info())
#891 rows, 714 missing values for Age, 2 missing values for Embarked
#drop of the useless columns
cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)

#we convert the Pclass, Sex, Embarked to columns in pandas and drop them after conversion.
#on créer une liste de dummies
dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
#dummies est une liste de 3 éléments * 891 éléments
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies),axis=1)
#Now that we converted Pclass, Sex, Embarked values into columns, 
#we drop the redundant same columns from the dataframe
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
#interpolate the missing values
df['Age'] = df['Age'].interpolate()

#######################################################################
#Start of the machine learning
X = df.values
y = df['Survived'].values
X = np.delete(X,1,axis=1)   #drop of the survived column
#we are ready with X and y, lets split the dataset for 70% Training and  30% test set using scikit cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state=42)

#Decision Tree Classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
print('Decision Tree Classifier accuracy: ', 100 * clf.score(X_test,y_test).round(3), '%')
print(100 * clf.feature_importances_.round(3))
#his output shows that second element in array 0.111,  “Fare” has 11% importance, 
#the last 5 shows 51% which is ‘Females’. Very interesting! yes the large number 
#of survivors in titanic are women and children.

#on améliore  le score de l'arbre de décision avec random forest
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit (X_train, y_train)
print('Random forest accuracy: ', 100 * clf.score(X_test,y_test).round(3), '%')

#on améliore le score de l'arbre de décision avec la descente de gradient
clf = ensemble.GradientBoostingClassifier()
clf.fit (X_train, y_train)
print('GradientBoostingClassifier accuracy: ', 100 * clf.score(X_test,y_test).round(3), '%')

#on améliore le score de l'arbre de décision avec la descente de gradient
clf = ensemble.GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train,y_train)
print('GradientBoostingClassifier accuracy: ', 100 * clf.score(X_test,y_test).round(3), '%')


###########################################################################
#determination of the y_test
df = pd.read_csv('titanic_test.csv')
print(df.info())
cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)
#creation of the dummies
dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies),axis=1)
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
#interpolate the missing values
df['Age'] = df['Age'].interpolate()
df['Fare'] = df['Fare'].interpolate()
#Start of the machine learning
X = df.values
X = np.delete(X,1,axis=1)   #drop of the survived column
y_test = clf.predict(df)

#upload of the data
df_y = pd.DataFrame({'Survived': y_test}, columns = ['Survived'])
results = pd.concat([df['PassengerId'], df_y], axis=1)
results.to_csv('Titanic_submissions.csv', index=False)