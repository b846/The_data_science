 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 08:51:58 2019

@author: b

## 0 Introduction
### 0.1 Libraries
### 0.2 Loading the Dataset
### 0.3 Functions
### 0.4 Overview
General information about the features
### 0.5 normality, linearity test
### 0.6 Correlation between the features
## 1 Feature selection
### 1.1 Drop of the columns with too much missing values
### 1.2 Dealing with missings values
Dealing with the outliers
Categorical to numerical data
### Distribution of the features
### 1.6 High correlation filter
### 1.7 importance of the features
## 2 Modelisation
### 2.1 Scores of different models
### Score of one model with differents inputs
### 2.2 Feature importances
### 2.3 Evolution of the score with the number of feature decreasng
"""


########################################################################
## 0 Introduction
### 0.1 Libraries
import numpy as np                   # mathematical operations
import pandas as pd                  # a powerful data analysis and manipulation library for Python
import matplotlib.pyplot as plt      # Matplotlib is a Python 2D plotting library which produces publication quality figures
import seaborn as sns                # statistical data visualization
sns.set(style="darkgrid")
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold   #Feature selector that removes all low-variance features.
from sklearn.preprocessing import StandardScaler
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
pd.options.display.max_rows = 4
graph_size = (10,8)
ratio_mv = 0.6


# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')


## 0.2. Loading the Dataset
df_train = pd.read_csv('house_price_train.csv')
df_test = pd.read_csv('house_price_test.csv')
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
col_Id, col_Target = 'Id', 'SalePrice'
df_train_target = pd.concat([df_train[col_Id] ,df_train[col_Target]], axis=1)
nb_row_train = df_train.shape[0]

## 0.3 Functions
def graph(group_x, df=df_all): 
    type1 = type(df_all[group_x][0])
    nb_x = len(df[group_x].unique())
    if type1 == np.int64 and nb_x < 15:
        # quantitative_discrète', Diagramme en secteurs
        df[group_x].value_counts(normalize=True).plot(kind='pie')
        plt.axis('equal')
    elif type1 == np.int64 and nb_x >= 15:
        sns.distplot(df[group_x], kde=False)
    elif type1 == np.float64:
        # 'quantitative_continue', histogramme
        # pour aller plus loin, voir le nombre optimal de classes pour l'agrégation : règle de Sturges k = 1 + log2(n)
        df[group_x].hist(density=True)
    else:
        #'qualitative_ordinale', diagramme en baton
        df[group_x].value_counts(normalize=True).plot(kind='bar')
    plt.title(group_x)
    plt.show()

def type2(group_x, df=df_all):
    #donne le type de variable
    type1 = type(df[group_x].dropna().iloc[0])
    if type1 == np.int64:
        return('quantitative_discrète')
    elif type1 == np.float64:
        return('quantitative_continue')
    else:
        return('qualitative_ordinale')


"""
group_x, group_y = 'LotArea', 'SalePrice'
def graph_corr(group_x, group_y=col_Target, df=df_all): 
     type2_x, type2_y = type2(group_x), type2(group_y)
    if type2_x[:12] == 'quantitative' and type2_y[:12] == 'quantitative':
        plt.scatter(df[group_x], df[group_y])
    elif type2_x[:12] == 'qualitative_' and type2_y[:12] == 'qualitative_':
        tab_contingence = pd.crosstab(x, y)
        sns.heatmap(tab_contingence)
    elif type2_x[:12] == 'qualitative_' and type2_y[:12] == 'quantitative':
        sns.barplot(x=group_x, y=group_y, data=df)
    elif type2_x[:12] == 'quantitative' and type2_y[:12] == 'qualitative_':
        sns.barplot(x=group_x, y=group_y, data=df, orient='h')
    plt.title('Correlation between {} and {}'.format(group_x, group_y))
    plt.xlabel(group_x)
    plt.ylabel(group_y)
    plt.legend()
    plt.show()
    """
    

### 0.4. Overview
print('------------------------------df_train.info()------------------------------')
print(df_train.info(), '\n')
print(df_train.sample(8), '\n')
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


#Information stored in a DataFrame
df_all_info = pd.DataFrame(index=df_all.columns, 
                columns=['nunique', 'type1', 'type2', 'Missing values', 'Variance', 
                         'Importance', 'Reg Linéaire', 'Reg Exp', 'Reg Log', 'Reg polynomiale'])
#value count
df_all_info['nunique'] = df_all.nunique()
#type
for col in df_all.columns:
    type1 = type(df_all[col][0])
    df_all_info['type1'][col] = type1
    if type1 == np.int64:
        df_all_info['type2'][col] = 'quantitative_discrète'
    elif type1 == np.float64:
        df_all_info['type2'][col] = 'quantitative_continue'
    else:
        df_all_info['type2'][col] = 'qualitative_ordinale'
df_all_info['type2']['Id'] = 'qualitative_nominale'
df_all_info['Missing values'] = df_all.isnull().sum()
# variance
#df_all_info['Variance'] = df_all_num.var().round(1)
    
print(df_all_info['type1'].value_counts())
print(df_all_info['type2'].value_counts())



"""
# GRAPH
for col in df_all.columns:
    if df_all_info['type2'][col] == 'quantitative_continue':
        graph(col)

for col in df_all.columns:
    if df_all_info['type2'][col] == 'quantitative_discrète':
        graph(col)
        
for col in df_all.columns:
    if df_all_info['type2'][col] == 'qualitative_ordinale':
        graph(col)

for col in df_all.columns:
    if df_all_info['type2'][col] == 'quantitative_continue':
        graph_corr(col)

for col in df_all.columns:
    if df_all_info['type2'][col] == 'quantitative_discrète':
        graph_corr(col)
        
for col in df_all.columns:
    if df_all_info['type2'][col] == 'qualitative_ordinale':
        graph_corr(col)
"""


# dealing with missing values
sns.set_style("whitegrid")
missing = df_all.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

"""
-on remplit par la valeur la plus fréquente
-on remplit par une nouvelle valeur, comme 0
-on remplit par la médiane d'nu groupe avec des similarités
"""

#Régression linéaire
for col in df_all.columns:
    nb_x = len(df_train[col].unique())
    type_x = type(df_train[col][0])
    if ((type_x == np.int64) or (type_x == np.float64)) and col != col_Target:
        df = pd.concat([df_train[col], df_train[col_Target]], axis=1).dropna(axis=0)
        x_matrix, y = df[col].values.reshape(-1,1), df[col_Target]
        reg = LinearRegression()         # création d'une classe  linearRegression
        reg.fit(x_matrix, y)
        df_all_info['Reg Linéaire'][col] = reg.score(x_matrix, y)




#test normality
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 10
normal = pd.DataFrame(df_train[quantitative])
normal = normal.apply(test_normality)
print(not normal.any())         #none of the features has a normal distribution


#This will run SciPy's normal test and print the results including a p representing A 2-sided chi 
#squared probability for the hypothesis test. If the p value is less than our alpha (significance 
#value), we can reject the hypothesis that this sample data is normally distributed. 
#If greater, we cannot reject the null hypothesis and must conclude the data is normally distributed.  
print(stats.normaltest(df_train[col_Target]))
graph(col_Target)

# Select the columns with too much missing values
col_mv = df_train.isnull().sum()/len(df_train)
# saving column names in a variable
col_with_too_much_mv = []
for i in range(0,len(df_train.columns)):
    if col_mv[i] >= ratio_mv:
        #setting the threshold as 60%
        col_with_too_much_mv.append(col_mv.index[i])



#Distribution


#### 2.2. Correlation between the columns
#sns.heatmap:(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, 
#sns.heatmap: fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, 
#sns.heatmap: square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
plt.figure(figsize=(12,10))
corr_matrix = df_all.corr().abs()
sns.heatmap(corr_matrix, annot=False, square=True, cmap='coolwarm')
plt.title('Features correlation')
plt.show()


# Select upper triangle of correlation matrix
upper_corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
col_with_high_correlation = [col for col in upper_corr_matrix.columns if any(upper_corr_matrix[col] > 0.95)]

#Plot an heatmap (arborescence)
sns.clustermap(corr_matrix, cmap='mako')

#Spearman correlation
def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
for q in qualitative:  
    encode(df_train, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)


#Dealing with missing values
#mode will return the most common data point
#-replace by the most_frequent value
#-replace by an similar value found by group
#-replace by a new value
for feature in df_all.columns:
    if df_all[feature].isnull().sum() > 0:    
        most_frequent_value = mode(df_all[feature])
        df_all[feature] = df_all[feature].fillna(most_frequent_value)




#categorical to numerical data
df_all_num = df_all.copy()
# all data are now numericals
print(df_all.info())
print(df_all_num.info())


#Creation of dummies
for feature in df_all.columns:
    df_all_num[feature] = LabelEncoder().fit_transform(df_all[feature].fillna(''))

[df_train_dummies, df_test_dummies] = df_all_num.loc[:nb_row_train], df_all_num.loc[nb_row_train+1:]



#Get dumies
df_all_dummies = df_all_num[qualitative]
for feature in quantitative:
    x_dummies = pd.get_dummies(df_all_num[feature], prefix=feature)
    df_all_dummies = pd.concat([df_all_dummies, x_dummies], axis=1)
[df_train_num, df_test_num] = df_all_num.loc[:nb_row_train], df_all_num.loc[nb_row_train+1:]

# Feature selection
# removing quasi-constant Feature
# Create VarianceThreshold object with a variance with a threshold of 0.1
# Passing a value of zero for the parameter threshold will filter all the features with zero variance
constant_filter = VarianceThreshold(threshold=.1)       #Feature selector that removes all low-variance features.
# Conduct variance thresholding
X_high_variance = constant_filter.fit_transform(df_train_num.drop([col_Target, col_Id], axis=1))
# View first five rows with features with variances above threshold
X_high_variance[0:5]
# remove some of the features
constant_filter.fit(df_train_num)
constant_filter.get_support()
print('non constant feature: ', len(df_train.columns[constant_filter.get_support()]), 'sur', len(df_train.columns))

"""
# To remove the constant feature from data, we can use transform
df_train_num_constant_filter = constant_filter.transform(df_train)
print(df_train_num_constant_filter.shape)
"""

# Feature selection
# Removing Correlated Features
correlated_features = set()             #empty object
for i in range(len(corr_matrix .columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            colname = corr_matrix.columns[i]
            correlated_features.add(colname)
print('Number of correlated features: ', len(correlated_features))
print(correlated_features)
# Choices that we have
# we can remove the correlated variables
# on peut mélanger ces variables avec d'autres variables




# Standardization of the quantitative feature


##################################################################
## 3. Machine Learning
### 3.1 Importation of models
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor


### 3.2 Functions used for the models
def prep_model(b_standardized, b_dummy, b_importance):
    # Function used to prepare the variables used for the function modelisation

    if b_dummy:
        df = df_train_dummies
    else:
        df = df_train_num
    nb_col, scores, list_importances = len(df.columns), [], []
    y = df[col_Target]
    
    if col_Id in df.columns:
        df = df.drop(col_Id, axis=1)
    if col_Target in df.columns:
        df = df.drop(col_Target, axis=1)
    
    if b_standardized:
        array = StandardScaler().fit_transform(df)
        df = pd.DataFrame(array, columns=df.columns)
    x = df
    
    # determine the number of folds
    if len(df) > 1000:
        nb_Fold = 1
    else:
        nb_Fold = 5
    
    if b_importance:
        #importances: contains the importances of the features
        nb_col = len(df.columns)
        list_importances = pd.DataFrame(np.zeros((nb_col, nb_Fold)), columns=['Fold_{}'.format(i) for i in range(1, nb_Fold + 1)], index=df.columns)
    
    return(x, y, list_importances, scores, nb_Fold)


def train_model(x, y, model, list_importances, scores, nb_Fold, b_importance):
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

    model, score, list_importances = train_model(x, y, model, list_importances, scores, nb_Fold, b_importance)
    
    #Return of the function
    if df_prediction is None:
        return(model, score, list_importances)
    else:
        submission_df = pd.DataFrame(columns=[col_Id, col_Target])
        submission_df[col_Id] = df_test[col_Id]
        submission_df[col_Target] = model.predict(x_test)
        return(submission_df)


#drop of the column

x_train = df_all_num.drop([col_Id, col_Target], axis=1)
[df_train_num, df_test_num] = df_all_num.loc[:nb_row_train], df_all_num.loc[nb_row_train:]


modelisation(model=XGBRegressor())

### 3.3 score of different model with different parameters
modelisation(model=XGBRegressor())
modelisation(model=linear_model.LinearRegression())
modelisation(model=linear_model.Lasso(alpha=0.1))
modelisation(model=RandomForestRegressor())
modelisation(model=GradientBoostingRegressor())
modelisation(model=ExtraTreesRegressor())
modelisation(model=BaggingRegressor())
modelisation(model=AdaBoostRegressor())


all_models = [XGBRegressor(),
              linear_model.LinearRegression(),
              linear_model.Lasso(alpha=0.1),
              RandomForestRegressor(),
              GradientBoostingRegressor(),
              ExtraTreesRegressor(),
              BaggingRegressor(),
              AdaBoostRegressor()]

list_score_models = pd.DataFrame(np.zeros((5, len(all_models))), columns = [
        'XGBRegressor',
        'linear_model.LinearRegression',
        'linear_model.Lasso(alpha=0.1)',
        'RandomForestRegressor',
        'GradientBoostingRegressor',
        'ExtraTreesRegressor',
        'BaggingRegressor',
        'AdaBoostRegressor'],
            index=['num not scaled', 'dummies not scaled', 'num scaled', 'dummies not scaled', 'time'])

for col, model_variant in enumerate(all_models):
    list_score_models.iloc[0, col] = modelisation(model=model_variant, b_standardized=False, b_dummy=False)[1]
    list_score_models.iloc[1, col] = modelisation(model=model_variant, b_standardized=False, b_dummy=True)[1]
    list_score_models.iloc[2, col] = modelisation(model=model_variant, b_standardized=True, b_dummy=False)[1]
    list_score_models.iloc[3, col] = modelisation(model=model_variant, b_standardized=True, b_dummy=True)[1]

print(list_score_models)
#now we need to select one model
#it seems that the model with dummies is very slow, and not so efficient

#Get the output
model, score, list_importances = modelisation(model=RandomForestRegressor(), b_standardized=False, b_dummy=False, b_importance=True)

#predictions
#Préparation du jeu de prédiction
modelisation(model=GradientBoostingRegressor(), df_prediction=x_test)


#select the important features
from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(RandomForestRegressor())
x = df_train_num.drop(col_Target, axis=1)
y = df_train_num[col_Target]
selector.fit(x, y)
#estimator_: The base estimator from which the transformer is built. 
selector.estimator_
#The threshold value to use for feature selection. Features whose importance is greater or equal 
#are kept while the others are discarded.
selector.threshold_
#get_support: Get a mask, or integer index, of the features selected
selector.get_support()
# 	transform: Reduce X to the selected features.
selector.transform(x)
#There are 9 important features for our model

#importance
len(x_train.columns)
importances = pd.DataFrame(model.feature_importances_)
importances.index = x_train.columns
df_all_info.loc[importances.index, 'Importance'] = model.feature_importances_

df_all_info.Importance.fillna(0, inplace=True)
df_all_info.sort_values(by='Importance', inplace=True, ascending=False)
print(df_all_info['Importance'])

#Graphe of the importances
plt.figure(figsize=(15, 20))
sns.barplot(x='Importance', y=df_all_info.index, data=df_all_info)
plt.show()

#show the evolution of the score, with the number of feature decreasing
nb_x = 81
scores = []
for i in range(nb_x):
    list_x = df_all_info.loc[:, 'Importance'].iloc[:i+1]
    x, y = df_train_num.drop([col_Target], axis=1), df_train_num.SalePrice
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=SEED)
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))


df_train_num.info()

#retour des valeurs
submission_df.to_csv('house_Price_submissions.csv', header=True, index=False)

