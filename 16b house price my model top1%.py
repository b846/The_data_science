#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:39:15 2020

@author: b
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p  #Compute the Box-Cox transformation of 1 + x.
from scipy.stats import boxcox_normmax  #Compute optimal Box-Cox transform parameter for input data.
import scipy.stats as statsmlxtend
import scipy.stats as stats
import seaborn as sns

#Models
import sklearn.linear_model as linear_model
from sklearn.linear_model import ElasticNetCV  #Model specific cross-validation, Tuning the hyper-parameters of an estimator
from sklearn.linear_model import LassoCV  #Linear model iterative fitting along a regularization path.
from sklearn.linear_model import RidgeCV #Ridge regression with built-in cross-validation.
from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosting for regression.
from sklearn.svm import SVR #Support Vector Machine
from xgboost import XGBRegressor #XGBoost stands for “Extreme Gradient Boosting”, supervised learning
from mlxtend.regressor import StackingCVRegressor

#The sklearn.decomposition module includes matrix decomposition algorithms, including among others PCA, NMF or ICA. 
#Most of the algorithms of this module can be regarded as dimensionality reduction techniques.
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms and estimators.
from sklearn.pipeline import make_pipeline
#RobustScaler: Scale features using statistics that are robust to outliers.
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold #K-Folds cross-validator
from sklearn.model_selection import  cross_val_score #Evaluate a score by cross-validation
from sklearn.metrics import mean_squared_error #Mean squared error regression loss
#Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
#Effective in high dimensional spaces.from sklearn.manifold import TSNE
from sklearn.manifold import TSNE #t-distributed Stochastic Neighbor Embedding. Mapping of words into numerical vector spaces







import os
os.chdir ('/home/b/Documents/Python/Data')

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
train = pd.read_csv('house_price_train.csv')
test = pd.read_csv('house_price_test.csv')
print ("Data is loaded!")

print ("Train: ",train.shape[0],"sales, and ",train.shape[1],"features")
print ("Test: ",test.shape[0],"sales, and ",test.shape[1],"features")
print(train.head())
print(test.head())

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

#missing values
sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()



#19 attributes have missing values, 5 over 50% of all data. 
#Most of times NA means lack of subject described by attribute, like missing pool, fence, no garage and basement.

#Graphe of the targets
y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)
#It is apparent that SalePrice doesn't follow normal distribution, so before performing regression it has to be transformed. 
#While log transformation does pretty good job, best fit is unbounded Johnson distribution.

#test of normality
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
normal = pd.DataFrame(train[quantitative])
normal = normal.apply(test_normality)
print("normal test: ", not normal.any())
#Also none of quantitative variables has normal distribution so these should be transformed as well.

#Correlation
"""
Spearman correlation is better to work with in this case because it picks up relationships between variables 
even when they are nonlinear. OverallQual is main criterion in establishing house price. Neighborhood has big influence,
 partially it has some intrisinc value in itself, but also houses in certain regions tend to share same characteristics 
 (confunding) what causes similar valuations.
"""
# Numérisation des variables qualitatives
# Nous allons ajouter des colonnes numérisées à notre DataFrame
def encode(frame, feature):
    # Permet de numériser une colonne qualitative
    # Création d'un dictionnaire contenant les valeurs prises et leur médiane
    # pour une variable qualitative
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique() #liste des valeurs uniques
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice'] #target mean of each unique value
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1) 
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        #cat permet de parcourir toutes les valeurs uniques
        frame.loc[frame[feature] == cat, feature+'_E'] = o

# On ajoute les colonnes numérisées à notre DataFrame
qual_encoded = []
for q in qualitative:  
    encode(train, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)

def spearman(frame, features):
    # Procédure permettant de montrer l'importance des colonnes
    spr = pd.DataFrame()
    spr['feature'] = features #1ère col=nom des features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features] #Corrélation de spearman
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
features = quantitative + qual_encoded
spearman(train, features)

# Tableau montrant la corrélation entre les variables
plt.figure(1)
corr = train[quantitative+['SalePrice']].corr()
sns.heatmap(corr)
plt.figure(2)
corr = train[qual_encoded+['SalePrice']].corr()
sns.heatmap(corr)
plt.figure(3)
corr = pd.DataFrame(np.zeros([len(quantitative)+1, len(qual_encoded)+1]), index=quantitative+['SalePrice'], columns=qual_encoded+['SalePrice'])
for q1 in quantitative+['SalePrice']:
    for q2 in qual_encoded+['SalePrice']:
        corr.loc[q1, q2] = train[q1].corr(train[q2])
sns.heatmap(corr)


#Simple clustering
features = quantitative + qual_encoded
model = TSNE(n_components=2, random_state=0, perplexity=50) #model embedding - the mapping of words into numerical vector spaces
X = train[features].fillna(0.).values  #Contient les valeurs numérisées
tsne = model.fit_transform(X)  #Fit X into an embedded space and return that transformed output.

std = StandardScaler()
s = std.fit_transform(X)  #modèle pour standardiser
pca = PCA(n_components=30) #modèle pour classification, Linear dimensionality reduction using Singular Value Decomposition of the data 
pca.fit(s)
pc = pca.transform(s)  #Classification des données
kmeans = KMeans(n_clusters=5)  #Création de la méthode pour le clustering
kmeans.fit(pc) #Créatin de 5 groupes de cluster

fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
print(np.sum(pca.explained_variance_ratio_))


#Model
#Data pre-processing
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)  #On enlève les indices précédemment sélectionnés
train["SalePrice"] = np.log1p(train["SalePrice"])  #log(1 + x)
y = train['SalePrice'].reset_index(drop=True)

#Features
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
print(features.shape)
features.head()

#Dealing with missing values
# On les variables quantitatives discrètes en variables qualitatives ordinales
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
#On remplit les valeurs manquantes avec la valeur la plus fréquente
features['Functional'] = features['Functional'].fillna('Typ') 
features['Electrical'] = features['Electrical'].fillna("SBrkr") 
features['KitchenQual'] = features['KitchenQual'].fillna("TA") 
features["PoolQC"] = features["PoolQC"].fillna("None")
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# Les valeurs NaN seront remplacées par 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#On remplace tous les valeurs manquantes par None pour les objets
#None sera transformé en nombre, cela évitera de planter
objects = [] #liste des objets
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#On remplace toutes les valeurs manquantes des colonnes numériques par 0
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))

# On vérifie la normalité de chaque variable continue avec skew
#For normally distributed data, the skewness should be about zero. 
#For unimodal continuous distributions, a skewness value greater than zero means 
#that there is more weight in the right tail of the distribution. 
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5] #contient des colonnes qui ne ressemblent pas à une distribution normale
skew_index = high_skew.index

# On normalise les variables qui ne sont pas normalisées
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
    
#On
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])

# Les variables avec trop de 0 prennent la valeur 1 si elles sont >0, 0 sinon
# cela permet de simplifier le modèle
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Get dummies
# pd.get_dummies: Convert categorical variable into dummy/indicator variables.
features.shape
final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape

X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
X.shape, y.shape, X_sub.shape

# On enlève les outliers
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

#On enlève les colonnes avec trop de zéros
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)




X.shape, y.shape, X_sub.shape

kfolds = KFold(n_splits=10, shuffle=True, random_state=42) #Méthode pour éviter l'overfittng

def rmsle(y, y_pred):
    #retourne l'erreur entre y et y_pred
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    #retourne l'erreur d'nu modèle avec une cross validation
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)



# On créer des variants que nous allons parcourir
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

#make_pipeline is a composite estimator
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)





xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)





score = cv_rmse(ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )


# Début du fitting
print('START Fit')
print(datetime.now(), 'elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)
print(datetime.now(), 'lasso')
lasso_model_full_data = lasso.fit(X, y)
print(datetime.now(), 'ridge')
ridge_model_full_data = ridge.fit(X, y)
print(datetime.now(), 'svr')
svr_model_full_data = svr.fit(X, y)
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
print(datetime.now(), 'xgboost')
xgb_model_full_data = xgboost.fit(X, y)

#il faut que la somme des coefficients fasse 1
# Il faudrait trouver un moyen pour interpoler les prédictions
# et encore améliorer le modèle
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.15 * lasso_model_full_data.predict(X)) + \
            (0.15 * ridge_model_full_data.predict(X)) + \
            (0.15 * svr_model_full_data.predict(X)) + \
            (0.15 * gbr_model_full_data.predict(X)) + \
            (0.3 * xgb_model_full_data.predict(X)))
            
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

print('Predict submission', datetime.now(),)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))

# this kernel gave a score 0.114
# let's up it by mixing with the top kernels

print('Blend with Top Kernals submissions', datetime.now(),)
sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')
sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')
sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')

submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))

# Brutal approach to deal with predictions close to outer range 
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("new_submission.csv", index=False)
print('Save submission', datetime.now(),)