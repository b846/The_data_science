#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:56:08 2019

@author: b
https://www.kaggle.com/goldens/titanic-on-the-top-with-a-simple-model
In this kernel I intend to use nested cross validation to choose between Random Forest and SVM.

Steps:
    1- Preprocessing and exploring
        1.1- Imports
        1.2- Types
        1.3 - Missing Values
        1.4 - Exploring
        1.5 - Feature Engineering
        1.6 - Prepare for models
    2- Nested Cross Validation
    3- Submission
"""

"""
x= df_train['Sex'].unique()
y= df_train['Sex'].value_counts()
plt.bar(x, y, edgecolor='white', label=y.index[i], bottom = y_bottom)
"""

##############################################################
#1- Preprocessing and exploring
#1.1 - Imports
# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import pandas as pd                  #a powerful data analysis and manipulation library for Python
import matplotlib.pyplot as plt      #Matplotlib is a Python 2D plotting library which produces publication quality figures
import seaborn as sns                # statistical data visualization
import numpy as np                   # mathematical operations
from time import time
# sklearn : Simple and efficient tools for predictive data analysis (Built on NumPy, SciPy, and matplotlib)
from sklearn.model_selection import cross_val_score   #Evaluate a score by cross-validation
#A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC           # C-Support Vector Classification.
from sklearn.preprocessing import StandardScaler    #Standardize features by removing the mean and scaling to unit variance
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor  #permet d'identifier la correlation entre les données
from sklearn.model_selection import GridSearchCV            #Exhaustive search over specified parameter values for an estimator.
import warnings #This is the base class of all warning category classes. It is a subclass of Exception.

#load the data np
 #raw_data_csv_np = np.loadtxt('Absenteeism-data.csv', delimiter = ',', dtype = 'str')
train=pd.read_csv('titanic_train.csv')
test=pd.read_csv('titanic_test.csv')
test2=pd.read_csv('titanic_test.csv')
pd.options.display.max_columns = 14    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20
titanic=pd.concat([train, test], sort=False)
len_train=train.shape[0]

#1.2 - Types
print(titanic.dtypes.sort_values())
print(titanic.select_dtypes(include='int').head())
print(titanic.select_dtypes(include='object').head())
print(titanic.select_dtypes(include='float').head())

#1.3 - Missing values
print(titanic.isnull().sum()[titanic.isnull().sum()>0])
train.Fare=train.Fare.fillna(train.Fare.mean())
test.Fare=test.Fare.fillna(train.Fare.mean())
train.Cabin=train.Cabin.fillna("unknow")
test.Cabin=test.Cabin.fillna("unknow")
train.Embarked=train.Embarked.fillna(train.Embarked.mode()[0])  #return the highest frequency value in a serie
test.Embarked=test.Embarked.fillna(train.Embarked.mode()[0])
#on applique la fct lambda aux noms, afin d'avoir Mr, Mrs, Miss, Master en title
train['title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
print(train['title'].unique())
print(test['title'].unique())
#on regroupe les données afin d'être fiable
newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}
train['title']=train.title.map(newtitles)
test['title']=test.title.map(newtitles)
#on affiche ici la moyenne des âges par titre et par sexe
train.groupby(['title','Sex']).Age.mean()
#on groupe les ages pour améliorer la fiabilité
def newage (cols):
    title=cols[0]
    Sex=cols[1]
    Age=cols[2]
    if pd.isnull(Age):
        if title=='Master' and Sex=="male":
            return 4.57
        elif title=='Miss' and Sex=='female':
            return 21.8
        elif title=='Mr' and Sex=='male': 
            return 32.37
        elif title=='Mrs' and Sex=='female':
            return 35.72
        elif title=='Officer' and Sex=='female':
            return 49
        elif title=='Officer' and Sex=='male':
            return 46.56
        elif title=='Royalty' and Sex=='female':
            return 40.50
        else:
            return 42.33
    else:
        return Age
train.Age=train[['title','Sex','Age']].apply(newage, axis=1)
test.Age=test[['title','Sex','Age']].apply(newage, axis=1)

#1.4 - Exploring¶
#on affiche des graphes pour visualiser les données
#sns.barplot: Show point estimates and confidence intervals as rectangular bars.
warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(3,3,1)
sns.barplot('Pclass','Survived',data=train)     # survived = f(Pclass)
plt.subplot(3,3,2)
sns.barplot('SibSp','Survived',data=train)      # survived = f(SbSp)
plt.subplot(3,3,3)
sns.barplot('Parch','Survived',data=train)      # survived = f(Parch)
plt.subplot(3,3,4)
sns.barplot('Sex','Survived',data=train)        # survived = f(sex)
plt.subplot(3,3,5)
sns.barplot('Ticket','Survived',data=train)     # survived = f(sex), graphe non lisible
plt.subplot(3,3,6)
sns.barplot('Cabin','Survived',data=train)      # survived = f(cabin), graphe non lisible
plt.subplot(3,3,7)
sns.barplot('Embarked','Survived',data=train)   # survived = f(embarked)
plt.subplot(3,3,8)
sns.distplot(train[train.Survived==1].Age, color='green', label='Survived', kde=False)
sns.distplot(train[train.Survived==0].Age, color='orange', label='Not survived', kde=False)
plt.subplot(3,3,9)
sns.distplot(train[train.Survived==1].Fare, color='green', kde=False)
sns.distplot(train[train.Survived==0].Fare, color='orange', kde=False)

#Interpretation
#SibSp and Parch don't seem to have a clear relationship with the target, 
#so put them together can be a good idea. For Ticket and Cabin a good strategie 
#can be count the number of caracteres.


#1.5 Feature Engineering
#on additionne dans 'Relative' le nb de parents et d'animal
train['Relatives']=train.SibSp+train.Parch
test['Relatives']=test.SibSp+test.Parch
#on calcule la longueur des tickets et des cabines
train['Ticket2']=train.Ticket.apply(lambda x : len(x))
test['Ticket2']=test.Ticket.apply(lambda x : len(x))
train['Cabin2']=train.Cabin.apply(lambda x : len(x))
test['Cabin2']=test.Cabin.apply(lambda x : len(x))
#on récupère le 1er mot
train['Name2']=train.Name.apply(lambda x: x.split(',')[0].strip())
test['Name2']=test.Name.apply(lambda x: x.split(',')[0].strip())
print('Nb de noms différents: ', len(train.Name2.unique()))

#on affiche des graphes
warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(3,3,1)
sns.barplot('Relatives','Survived',data=train)  # survived = f(Relatives), Relatives=nb de d'enfants et de parents
plt.subplot(3,3,2)
sns.barplot('Ticket2','Survived',data=train)    # survived = f(Ticket2), Ticket2=longueur du str ticket
plt.subplot(3,3,3)
sns.barplot('Cabin2','Survived',data=train)     # survived = f(Cabin2), Cabin2==longueur du str cabine


#1.6 - Prepare for model
#droping features I won't use in model
#train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin']
#on enlève ces colonnes car elles sont inutiles pour une modélisation
train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
titanic=pd.concat([train, test], sort=False)
titanic=pd.get_dummies(titanic)     #on split les données en dummies
print(titanic.shape)
train=titanic[:len_train]
test=titanic[len_train:]
# Turning categorical into numerical
train.Survived=train.Survived.astype('int')     # cast a pandas object to a specified dtype
print(train.info())
#on enlève 'Survived'
xtrain=train.drop("Survived",axis=1)
ytrain=train['Survived']
xtest=test.drop("Survived", axis=1)


#2 - Nested Cross Validation
#Random Forest
RF=RandomForestClassifier(random_state=1)       #classe random forest classifier, meta estimator
PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]
#class GridSearchCV, Exhaustive search over specified parameter values for an estimator.
#cv : int, cross-validation generator or an iterable, optional. 
#cv: Determines the cross-validation splitting strategy.
GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)
#cross_val_score: Evaluate a score by cross-validation
scores_rf=cross_val_score(GSRF,xtrain,ytrain,scoring='accuracy',cv=5)
print(scores_rf)
print(scores_rf.mean)

#Random Forest
RF=RandomForestClassifier(random_state=1)
scores_rf1=cross_val_score(RF,xtrain,ytrain,scoring='accuracy',cv=5)
print(np.mean(scores_rf1))
RF.fit(xtrain, ytrain)
#Check the feature Importance
importances=RF.feature_importances_
feature_importances=pd.Series(importances, index=xtrain.columns).sort_values(ascending=False)
sns.barplot(x= feature_importances[0:10] , y= feature_importances.index[0:10])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()

"""
#Partial Dependence Plots
#Partial dependence plots show the dependence between the target function 2 and a set of ‘target’ features
print("Training GradientBoostingRegressor...")
tic = time()
est = HistGradientBoostingRegressor()
est.fit(xtrain, ytrain)
print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(est.score(xtest, ytest)))
"""

#SVM
#make_pipeline: Construct a Pipeline from the given estimators.
#class SVC(sklearn.svm.base.BaseSVC): C-Support Vector Classification.
svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)
print(np.mean(scores_svm))


#3 - Submission
model=GSSVM.fit(xtrain, ytrain)
pred=model.predict(xtest)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('titanic_submission.csv', index=False)