#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:35:57 2019
je vais ajouter le tableau de comparaisons des bonnes prédictions
et ensuite je ferais un clustering avec ces données

@author: b
## 0 Introduction
### 0.1 Libraries
### 0.2 Loading the Dataset
### 0.3 Functions
### 0.4 Overview
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
from sklearn.metrics import confusion_matrix
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
graph_size = (10,8)

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')


## 0.2. Loading the Dataset
df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
col_Id, col_Target = 'PassengerId', 'Survived'
df_train_target = pd.concat([df_train[col_Id] ,df_train[col_Target]], axis=1)
nb_row_train = df_train.shape[0]

## 0.3 Functions
def graph(group_x, group_y=col_Target, df=df_all):    
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
        sns.distplot(df[surv][group_x].dropna(axis=0), label=col_Target, hist=True, color='#2ecc71')
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

# Qualitative and quantitative
quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
if col_Target in quantitative:
    quantitative.remove(col_Target)
if col_Id in quantitative:
    quantitative.remove(col_Id)
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']

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
graph(group_x='Sex', group_y=col_Target, df=df_train)

### 1.3  Additional variables
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1


########################################################################
### 2. Feature Engineering
#### 2.1 Grouping values inside columns
# we group the decks with similarities
#df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
#df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
#df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
print(df_all['Deck'].value_counts())

#df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
#df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


#### 2.2. Correlation between the columns
#sns.heatmap:(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, 
#sns.heatmap: fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, 
#sns.heatmap: square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
print(df_all.info())
plt.figure(figsize=(12,10))
df=df_all.drop(['PassengerId', col_Target, 'Ticket'], axis=1)
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
columns_to_remove = ['Name', 'PassengerId', 'Ticket', 'Cabin', col_Target]
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
#DecisionTreeRegressor
class sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import HuberRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier


### 3.2 Functions used for the models
def prep_model(b_standardized, b_dummy, b_importance):
    # Function used to prepare the variables used for the function modelisation
    if b_dummy:
        df = df_train_dummies
    else:
        df = df_train_num
    nb_col, scores, list_importances = len(df.columns), [], []
    
    if col_Id in df.columns:
        df = df.drop(col_Id, axis=1)
    if col_Target in df.columns:
        df = df.drop(col_Target, axis=1)
    
    if b_standardized:
        array = StandardScaler().fit_transform(df)
        df = pd.DataFrame(array, columns=df.columns)
    x, y = df, df_train_target[col_Target]
    
    # determine the number of folds
    if len(df) > 1000:
        nb_Fold = 1
    else:
        nb_Fold = 5
    
    if b_importance:
        #importances: contains the importances of the features
        nb_col = len(df.columns), 5
        list_importances = pd.DataFrame(np.zeros((nb_col, nb_Fold)), columns=['Fold_{}'.format(i) for i in range(1, nb_Fold + 1)], index=df.columns)
    
    return(x, y, list_importances, scores, nb_Fold)


def train_model(x, y, list_importances, scores, nb_Fold, b_importance):
    # Function used to train the model
    # Creation of the mechanism to split the data
    if nb_Fold >2:
        #StratifiedKFold: Provides train/test indices to split data in train/test sets.
        skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
        #trn_idx: index of df_train; val_idx: index of df_test
        for fold, (trn_idx, val_idx) in enumerate(skf.split(x, y), 1):
            # Fitting the model with the train data
            model.fit(x.iloc[trn_idx], y.iloc[trn_idx])
            scores.append(model.score(x.iloc[val_idx], y.iloc[val_idx]))
            if b_importance:
                list_importances.iloc[:, fold-1] = model.feature_importances_
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=SEED)
        model.fit(x_train, y_train)
        if b_importance:
            list_importances.iloc[:, 0] = model.feature_importances_
    
    # Preparation of the return of the function
    if nb_Fold>2:
        score = 100 * np.mean(scores).round(3)
    else:
        score = 100 * model.score(x_test, y_test).round(3)
    
    if b_importance:
        list_importances = list_importances.mean(axis=1)

    return(model, score, list_importances)
    
def modelisation(model, b_standardized=False, b_dummy=False, b_importance=False, df_prediction=None):
    # function that will give the score of the model or the predicted values if df_prediction is not None
    # Preparation of the modelisation
    x, y, list_importances, scores, nb_Fold = prep_model(b_standardized, b_dummy, b_importance)

    model, score, list_importances = train_model(x, y, list_importances, scores, nb_Fold, b_importance)
    
    #Return of the function
    if df_prediction is None:
        return(model, score, list_importances)
    else:
        submission_df = pd.DataFrame(columns=[col_Id, col_Target])
        submission_df[col_Id] = df_train[col_Id]
        submission_df[col_Target] = model.predict(df_prediction)
        return(submission_df)


### 3.3 score of different model with different parameters
model=RandomForestClassifier()
modelisation(model=RandomForestClassifier())
modelisation(model=DecisionTreeRegressor())
modelisation(model=XGBRegressor())
modelisation(model=LogisticRegression())
modelisation(model=HuberRegressor())
modelisation(model=linear_model.LinearRegression())
modelisation(model=linear_model.Lasso(alpha=0.1))
modelisation(model=KNeighborsClassifier(n_neighbors=3))

all_models = [RandomForestClassifier(),
              DecisionTreeRegressor(),
              XGBRegressor(),
              LogisticRegression(),
              HuberRegressor(),
              linear_model.LinearRegression(),
              linear_model.Lasso(alpha=0.1),
              KNeighborsClassifier(n_neighbors=3)]



list_score_models = pd.DataFrame(np.zeros((4, len(all_models))), columns = [
        'RandomForestClassifier',
        'DecisionTreeRegressor',
        'XGBRegressor',
        'LogisticRegression',
        'HuberRegressor',
        'linear_model.LinearRegression',
        'linear_model.Lasso(alpha=0.1)',
        'KNeighborsClassifier(n_neighbors=3)'],
            index=['num not scaled', 'dummies not scaled', 'num scaled', 'dummies not scaled'])


model=RandomForestClassifier()
for col, model_variant in enumerate(all_models):
    list_score_models.iloc[0, col] = modelisation(model=model_variant, b_standardized=False, b_dummy=False)[1]
    list_score_models.iloc[1, col] = modelisation(model=model_variant, b_standardized=False, b_dummy=True)[1]
    list_score_models.iloc[2, col] = modelisation(model=model_variant, b_standardized=True, b_dummy=False)[1]
    list_score_models.iloc[3, col] = modelisation(model=model_variant, b_standardized=True, b_dummy=True)[1]

print(list_score_models)
#According to the scores of the models, RandomForestClassifier is the best for our case
#We will continue to improve our model with RandomForestClassifier

#output of modelization
model, score, importances = modelisation(model=RandomForestClassifier())

# Predictions
y_pred = modelisation(model=DecisionTreeRegressor(), df_prediction=df_train_num)



### 3.4 Score of the model chosen with different parameters
"""RandomForestClassifier
La complexité du modèle est gérée par deux paramètres :
 -max_depth, qui détermine le nombre max de feuilles dans l’arbre,
 -et le nombre minimales min_samples_splitd’observations requises pour rechercher une dichotomie.
forest = RandomForestClassifier(n_estimators=500,criterion=’gini’, max_depth=None,
min_samples_split=2, min_samples_leaf=1,max_features=’auto’, max_leaf_nodes=None,bootstrap=True, oob_score=True)
"""



"""
### 3.5 Histogram of the importance of each feature
#https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html?highlight=model%20feature_importances_
model, score, importances = modelisation(model=RandomForestClassifier())     #output of modelization
importances.sort_values(inplace=True, ascending=False)
plt.figure(figsize=graph_size)
plt.ylabel('Importances')
plt.title('Importances of the features')
plt.plot(importances)
plt.show()
"""

###  3.6 Confusion matrix
# A confusion matrix can measure the performance of the matrix
def performance_binaire(reel, pred):
    if reel == 0 and pred == 0:
        return('TN')
    elif reel == 1 and pred == 0:
        return('FP')
    elif reel == 0 and pred == 1:
        return('FN')
    elif reel == 1 and pred == 1:
        return('TP')


y_true = df_train[col_Target]
y_pred = model.predict(df_train_num)
print(confusion_matrix(y_true, y_pred))
df_train['Prediction'] = y_pred
df_train['Performance'] = None
df_train['Performance'] = df_train.apply(lambda x: performance_binaire(x[col_Target], x['Prediction']), axis=1)


### 3.7 Clustering
# on réalise un cluster avec la confusion matrix,
# cela permettra d'identifier des groupes de données
#ce qu'il me faudrait c'est les indices, ou éliminer les lignes inutiles
TN = df_train[df_train['Performance'] == 'TN']
FN = df_train[df_train['Performance'] == 'FN']
FP = df_train[df_train['Performance'] == 'FP']
TP = df_train[df_train['Performance'] == 'TP']
#Graphe
TN['Age'].hist()
plt.show()
FN['Age'].hist()
plt.show()
FP['Age'].hist()
plt.show()
TP['Age'].hist()
plt.show()

sns.boxplot(x="Age", y="Performance", data=df_train,
            whis="range", palette="vlag")
plt.show()
sns.boxplot(x="Fare", y="Performance", data=df_train,
            whis="range", palette="vlag")
plt.show()

#add a scatter plot group_x, group_y avec 4 couleurs
def graph_cluster(group_x, group_y):
    sns.scatterplot(x=group_x, y=group_y, data=TN, label='TN')
    sns.scatterplot(x=group_x, y=group_y, data=FN, label='FN')
    sns.scatterplot(x=group_x, y=group_y, data=FP, label='FP')
    sns.scatterplot(x=group_x, y=group_y, data=TP, label='TP')
    plt.legend()
    plt.show()

group_x, group_y = 'Age', 'Fare'
sns.scatterplot(x=group_x, y=group_y, data=TN)
graph_cluster(group_x, group_y)
             
########################################################################
## 4 Save of the model

submission_df = pd.DataFrame(columns=['PassengerId', col_Target])
submission_df['PassengerId'] = df_test['PassengerId']
#submission_df[col_Target] = np.where(y_pred>=0.5, 1,0)
#submission_df.to_csv('Titanic_submissions.csv', header=True, index=False)
