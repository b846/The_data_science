#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:35:57 2019

@author: b
## 0 Introduction
### 0.1 Libraries
### 0.2 Loading the Dataset
### 0.3 Functions
### 0.4 Overview
### 0.5 Histogram
## 1. Dealing with Missing Values
## 2. Feature Engineering
### 2.1 Grouping values inside columns
### 2.2. Correlation between the columns
### 2.3 Graphs
### 2.4 Categorical data to numerical data
### 2.5 Get dummies
## 3. Machine Learning
### 3.1 Importation of models
### 3.2 Functions used for the models
### 3.3 score of the models with different parameters
### 3.4 Gaphe of the models
### 3.5 Histogram of the importance of each feature
## 4. Save of the results
"""

########################################################################
## 0 Introduction
### 0.1 Libraries
import numpy as np                   # mathematical operations
import pandas as pd                  # a powerful data analysis and manipulation library for Python
import matplotlib.pyplot as plt      # Matplotlib is a Python 2D plotting library which produces publication quality figures
import seaborn as sns                # statistical data visualization
sns.set(style="darkgrid")

#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from statistics import mode

#Deal with the warnings
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# others parameters
SEED = 42
pd.options.display.max_columns = 20    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')


## 0.2. Loading the Dataset
df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
df_train_target = pd.concat([df_train['PassengerId'] ,df_train['Survived']], axis=1)


## 0.3 Functions
def graph(group_x, group_y='Survived', df=df_all):    
    #Ajouter une sécurité si l'un des paramètres n'est pas présent
    #Display a numerical or categorical graph for the data
    print(group_x, ': type=', type(df[group_x].unique()[1]), '; missing values=', df[group_x].isna().sum())
    print(df[group_x].value_counts())
    df = pd.concat([df[group_x], df[group_y]], axis=1).dropna(axis=0)

    #Display a graph of the data
    nb_x = len(df[group_x].unique())
    nb_y = len(df[group_y].unique())
    type_x = type(df[group_x].unique()[0])
    if ((type_x == np.int64) or (type_x == np.float64)) and nb_x >10:
        # display a graph with numerical data
        df = pd.concat([df[group_x], df[group_y]], axis=1).dropna(axis=0)
        surv = df[group_y] == 1
        sns.distplot(df[~surv][group_x], label='Not Survived', hist=True, color='#e74c3c')
        sns.distplot(df[surv][group_x].dropna(axis=0), label='Survived', hist=True, color='#2ecc71')
        plt.title('Distribution of {} in {}'.format(group_y, group_x), size=13, y=1)
        plt.legend()
        plt.show()
        
    elif nb_x <10 and nb_y <5:
        # display a graph with categorical data
        dummies_y = pd.get_dummies(df[group_y], prefix = group_y)
        xy = pd.concat([df[group_x], dummies_y.iloc[:,-nb_y:]], axis=1)
        y = xy.groupby([group_x]).sum().transpose()
        x = y.columns
        y_bottom = np.zeros(nb_x)
        for i in np.arange(start=0, stop=nb_y):
            plt.bar(x, y.iloc[i,:], edgecolor='white', label=y.index[i], bottom = y_bottom)
            y_bottom += y.iloc[i,:]
            plt.legend()
        plt.xlabel(group_x)
        plt.ylabel(group_y)
        plt.title('Passenger {} Distribution in {}'.format(group_y, group_x))
        plt.show()
    else:
        print()


def display_median_by_group(feature, group1, group2=None):
    # display the median of a df grouped
    df = pd.concat([df_train[feature], df_train[group1], df_train[group2]], axis=1).dropna(axis=0)
    if group2 is not None :
        data_grouped_with_median = df.groupby([group1, group2]).median()[feature]
        for g1 in df[group1].unique():
            for g2 in df[group2].unique():
                print('Median {} of {}={} and {}={} : {}'.format(feature, group1, g1, group2, g2, data_grouped_with_median[g1][g2]))
    else:
        data_grouped_with_median = df.groupby([group1]).median()[feature]
        for g1 in df_all[group1].unique():
            print('Median {} of {}={} : {}'.format(feature, group1, g1, data_grouped_with_median[g1]))
    print('Median {} of all passengers: {}'.format(feature, df[feature].median()))



### 0.4. Overview
print('------------------------------df_train.info()------------------------------')
print(df_train.info(), '\n')
print(df_train.sample(3), '\n')
print('------------------------------df_test.info()------------------------------')
print(df_test.info(), '\n')
print(df_test.sample(3), '\n')
#we can see the type of the data and the missing values

### 0.5 histogram
#on compte le nombre de variables numériques
#on affiche des histogrammes
""" 
# Automatisation des histogrammes numériques et non numériques
fig = plt.figure(figsize=(16,12))
for feat_idx in range(df_train.shape[1]-1):
    if (isinstance(df_train.iloc[:, feat_idx][0], float) or isinstance(df_train.iloc[:, feat_idx][0], np.int64)):
        ax = fig.add_subplot(3,4, (feat_idx+1))
        h = ax.hist(df_train.iloc[:, feat_idx], bins=50, color='steelblue', normed=True)
        ax.set_title(df_train.columns[feat_idx], fontsize=14)
plt.show()
"""

########################################################################
### 1. Dealing with Missing Values
#### 1.1. Age
graph('Age')
#we suggest replacing the missing values with the group medians
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
display_median_by_group(feature = 'Age', group1 = 'Sex', group2 = 'Pclass')
# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


#### 1.2. Fare
graph('Fare')
display_median_by_group(feature='Fare', group1='Pclass', group2='Embarked')
#we display the line with a missing value
print(df_all[df_all['Fare'].isnull()])
# Filling the missing values in Fare
most_frequent_value = mode(df_all['Fare'])
df_all['Fare'] = df_all['Fare'].fillna(most_frequent_value)


### 1.3. Embarked
graph('Embarked')
# Filling the missing values in Embarked with the most common value
df_all['Embarked'] = df_all['Embarked'].fillna('S')


#### 1.4. Cabin
graph('Cabin')
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
print(df_all['Deck'].value_counts())
graph(group_x='Deck', group_y='Pclass', df=df_all)


#### 1.2.4. Sex
graph(group_x='Sex', group_y='Survived', df=df_train)

### 1.3  Additional variables
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1


########################################################################
### 2. Feature Engineering
#### 2.1 Grouping values inside columns
# we group the decks with similarities
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
print(df_all['Deck'].value_counts())

df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


#### 2.2. Correlation between the columns
#sns.heatmap:(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, 
#sns.heatmap: fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, 
#sns.heatmap: square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
print(df_all.info())
plt.figure(figsize=(12,10))
df=df_all.drop(['PassengerId', 'Survived', 'Ticket'], axis=1)
sns.heatmap(df.corr(), annot=True, square=True, cmap='coolwarm')
plt.title('Features correlation')
plt.show()


#### 2.3 Graphs
graph('Parch')
graph('Embarked')
graph('Pclass')
graph('SibSp')
graph('Deck')
graph('Title')


### 2.4 columns to remove
print(df_all.columns)
columns_to_remove = ['Name', 'PassengerId', 'Ticket', 'Cabin', 'Survived']
df_all_num = df_all.drop(columns_to_remove, axis=1)


### 2.5 Categorical data to numerical data
# the object type and category type will be converted to numerical type with LabelEncoder
object_category_col = ['Embarked', 'Parch', 'Sex', 'SibSp', 'Deck', 'Title']
#sklearn.preprocessing.LabelEncoder: Encode target labels with value between 0 and n_classes-1.
for feature in object_category_col:        
        df_all_num[feature] = LabelEncoder().fit_transform(df_all[feature])
# all data are now numericals
print(df_all_num.info())
[df_train_num, df_test_num] = df_all_num.loc[:890], df_all_num.loc[891:]


### 2.6 Get dummies
#Get dummies for the categorical data
numerical_features = ['Age', 'Fare']
df_all_dummies = pd.concat([df_all['Age'], df_all['Fare']], axis=1)

categorical_features = ['Embarked', 'Pclass', 'Sex', 'Deck', 'Title', 'Is_Married']
for feature in categorical_features:
    x_dummies = pd.get_dummies(df_all_num[feature], prefix=feature)
    df_all_dummies = pd.concat([df_all_dummies, x_dummies], axis=1)

[df_train_dummies, df_test_dummies] = df_all_dummies.loc[:890], df_all_dummies.loc[891:]
print(df_all_dummies.info())
#the data are ready to be modelized



##################################################################
## 3. Machine Learning
### 3.1 Importation of models
"""
#RandomForestClassifier
La complexité du modèle est gérée par deux paramètres :
 -max_depth, qui détermine le nombre max de feuilles dans l’arbre,
 -et le nombre minimales min_samples_splitd’observations requises pourrechercher une dichotomie.
forest = RandomForestClassifier(n_estimators=500,criterion=’gini’, max_depth=None,
min_samples_split=2, min_samples_leaf=1,max_features=’auto’, max_leaf_nodes=None,bootstrap=True, oob_score=True)

#DecisionTreeRegressor
class sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


### 3.2 Functions used for the models
def modelisation(model=RandomForestClassifier(), df=df_train_num, max_Fold=15, standardized=False, df_prediction=None):
    # function that will give the score of the model or the predicted values if df_prediction is not None
    #StandardScaler: Standardize features by removing the mean and scaling to unit variance
    if standardized:
        array = StandardScaler().fit_transform(df)
        df = pd.DataFrame(array, columns=df.columns)
    x, y = df, df_train_target['Survived']
    scores = []      # list of the scores
    
    #importances: contains the importances of the features
    nb_col = len(df.columns)
    importances = pd.DataFrame(np.zeros((nb_col, max_Fold)), columns=['Fold_{}'.format(i) for i in range(1, max_Fold + 1)], index=df.columns)

    #train of the model
    # Creation of the mechanism to split the data
    if max_Fold > 1:
        #StratifiedKFold: Provides train/test indices to split data in train/test sets.
        skf = StratifiedKFold(n_splits=max_Fold, random_state=SEED, shuffle=True)
        #trn_idx: index of df_train; val_idx: index of df_test
        for fold, (trn_idx, val_idx) in enumerate(skf.split(x, y), 1):
            # Fitting the model with the train data
            model.fit(x.iloc[trn_idx], y.iloc[trn_idx])
            scores.append(model.score(x.iloc[val_idx], y.iloc[val_idx]))
            importances.iloc[:, fold-1] = model.feature_importances_
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=SEED)
        model.fit(x_train, y_train)
        importances.iloc[:, 0] = model.feature_importances_
    
    # DataFrame with the predictions
    submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
    submission_df['PassengerId'] = df_test['PassengerId']
    
    # return the score of the model if df_prediction is None
    # return the prediction of the model if df_prediction is not None
    if df_prediction is None and max_Fold >1:
        return(model, 100 * np.mean(scores).round(3), importances.mean(axis=1))
    elif df_prediction is None and max_Fold <= 1:
        return(model, 100 * model.score(x_test, y_test).round(3), importances.mean(axis=1))
    elif df_prediction is not None and max_Fold >1:
        predictions = model.predict(df_prediction)
        submission_df['Survived'] = np.where(predictions>0.5, 1,0)
        return(model.predict(df_prediction))
    elif df_prediction is not None and max_Fold <=1:
        predictions = model.predict(df_prediction)
        submission_df['Survived'] = np.where(predictions>0.5, 1,0)
        return(model.predict(df_prediction))


def graph_score_model(model=RandomForestClassifier(), max_fold=9, step_fold=3, nb_iteration=2, df=df_train_num, standardized=False):
    # display a graph which reprents the score of the model, depending of the number of folds
    # il faudrait expliciter un peu plus list_scores
    folds = np.arange(1,max_fold,step_fold)
    list_scores = np.zeros((nb_iteration,max_fold // step_fold))
    for j in np.arange(start=0, stop=nb_iteration, step=1):
        for i, fold in enumerate(folds):
            list_scores[j,i] = modelisation(model=RandomForestClassifier(), max_Fold=fold, df=df, standardized=standardized)[1]

    y = np.sum(list_scores, axis=0) / nb_iteration
    plt.plot(folds, y, label=type(model))



### 3.3 score of the models with different parameters
modelisation(model=RandomForestClassifier())
modelisation(model=DecisionTreeRegressor())
modelisation(model=XGBRegressor(), max_Fold=1)

modelisation(model=RandomForestClassifier(), standardized=True)
modelisation(model=DecisionTreeRegressor(), standardized=True)
modelisation(model=XGBRegressor(), standardized=True)

#output of modelization
model, score, importances = modelisation(model=RandomForestClassifier())

# Predictions
modelisation(model=RandomForestClassifier(), df_prediction=df_test_num)

# Graph
modelisation(model=RandomForestClassifier(), max_Fold=5, df=df_train_num, standardized=False)
graph_score_model(model=RandomForestClassifier())

### 3.4 Gaphe of the models
#max_fold=20, step_fold=2, nb_iteration=10, df=df_train_dummies, standardized=False
plt.figure(figsize=(12,10))
graph_score_model(model=RandomForestClassifier())
graph_score_model(model=DecisionTreeRegressor())
graph_score_model(model=XGBRegressor)
plt.xlabel('number of fold')
plt.ylabel('score of the model')
plt.title('Score of different models')
plt.legend()
plt.show()


### 3.5 Histogram of the importance of each feature
#https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html?highlight=model%20feature_importances_
model, score, importances = modelisation(model=RandomForestClassifier())     #output of modelization
importances.sort_values(inplace=True, ascending=False)
plt.plot(importances)
plt.show()

########################################################################
## 4 Save of the model
#submission_df.to_csv('Titanic_submissions.csv', header=True, index=False)
