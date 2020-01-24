#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:24:42 2020

@author: b


"""

#1.1- Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from scipy.stats import norm
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p  #Compute the Box-Cox transformation of 1 + x.
from scipy.stats import boxcox_normmax  #Compute optimal Box-Cox transform parameter for input data.
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
import os
#print(os.listdir('/home/b/Documents/Python/Data'))


# Option d'affichage
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 5    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 50
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
graph_size = (10,8)

#Load of the data
import os
os.chdir ('/home/b/Documents/Python/Data')
train = pd.read_csv('house_price_train.csv')
test = pd.read_csv('house_price_test.csv')
col_Id, col_Target = 'Id', 'SalePrice'
print ("Data is loaded!")

print ("Train: ",train.shape[0],"sales, and ",train.shape[1],"features")
print ("Test: ",test.shape[0],"sales, and ",test.shape[1],"features")
print(train.head())
print(test.head())
print (train.columns)
print(test.columns)
print(train.shape,test.shape)

#Save the 'Id' column
train_ID = train[col_Id]
test_ID = test[col_Id]

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Qualitative and quantitative
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
if col_Target in quantitative:
    quantitative.remove(col_Target)
if col_Id in quantitative:
    quantitative.remove(col_Id)
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

#2.2 Quiring the data
train.describe().transpose()

#3 The predicted variable - Sales price Skew & kurtosis analysis
"""
The predicted variable is probably the most important variable, 
therefore it should be inspected throughly.
It turns out models work better with symmetric gaussian distributions, 
therefore we want to get rid of the skewness by using log transformation.

SKEWNESS
skewness is a measure of the asymmetry of the probability distribution of a 
real-valued random variable about its mean. The skewness value can be positive 
or negative, or undefined. 

KURTOSIS
Mesure de la hauteur d'une distribution
"""

#3.1 Observing Sale price histogram¶
train[col_Target].describe()
sns.distplot(train[col_Target])
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

# 3.2 Transforming the target into a normal distribution
# Plot histogram and probability
fig = plt.figure(figsize=(13,4))
plt.subplot(1,2,1)
sns.distplot(train[col_Target] , fit=norm)
(mu, sigma) = norm.fit(train[col_Target])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
# stats.probplot: Generates a probability plot of sample data against the quantiles of a specified theoretical distribution
res = stats.probplot(train[col_Target], plot=plt)
plt.suptitle('Before transformation')
# we can see that we are close to the normal distribution

# Apply transformation in order to get closer to a normal distribution
train.SalePrice = np.log1p(train.SalePrice) #log(1+x)
# New prediction
y_train = train.SalePrice.values
y_train_orig = train.SalePrice


# Plot histogram and probability after transformation
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train[col_Target] , fit=norm) #histogram of the target
(mu, sigma) = norm.fit(train[col_Target])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.suptitle('After transformation')

# Concatenate train and test
# y_train_orig = train.SalePrice
# train.drop("SalePrice", axis = 1, inplace = True)
data_features = pd.concat((train, test)).reset_index(drop=True)
print(data_features.shape)
# print(train.SalePrice)


# 4 Missing data
# 4.1 Locating missing data
#missing data percent plot
total = data_features.isnull().sum().sort_values(ascending=False)
percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)


# 4.2 replacing the missing data
#String values
#For numbers that have no significance and should actually be strings

str_vars = ['MSSubClass','YrSold','MoSold']
for var in str_vars:
    data_features[var] = data_features[var].apply(str)

# For the data with less than 10 mv, we will substitute the most common value
# Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
common_vars = missing_data[missing_data.Percent < 0.1][missing_data.Total > 0].index

for var in common_vars:
    data_features[var].fillna(data_features[var].mode()[0], inplace=True)

# For the categorical features with a majority of  NA, we will substitute NaN by None
data_features.head().transpose()

for var in missing_data[missing_data.Percent > 0.5].index:
    if var in qualitative:
        data_features[var].fillna('None', inplace=True)
    if var in quantitative:
        data_features[var].fillna(0, inplace=True)


#Only 2 features have between 10% and 50% of missing values
col_medium_mv = missing_data[missing_data.Percent >= 0.1][missing_data.Percent <= 0.5].index
print(missing_data[missing_data.Percent >= 0.1][missing_data.Percent <= 0.5])
print(train.columns)
# We will fill the 2 features with the median of a relevant group
train.loc[:6,['FireplaceQu', 'Neighborhood', 'Fireplaces', 'LotFrontage']]

# We will fill the col_mv by the most common value of col_mv of the group1
col_mv, group1 ='LotFrontage', 'Neighborhood'

for group in data_features[group1].unique():
    most_frequent_value_group = data_features[data_features[group1] == group][col_mv].dropna().mode()[0]
    data_features[data_features[group1] == group][col_mv].fillna(most_frequent_value_group, inplace = True)

data_features.LotFrontage.fillna(data_features.LotFrontage.median(), inplace= True)
data_features.FireplaceQu.fillna('None', inplace= True)


#5 Numerical and Categorial features
#5.1 Splitting the data into categorial and numerical features

# Differentiate numerical features (minus the target) and categorical features
categorical_features = data_features.select_dtypes(include=['object']).columns
print(categorical_features)
numerical_features = data_features.select_dtypes(exclude = ["object"]).columns
print(numerical_features)

print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
feat_num = data_features[numerical_features]
feat_cat = data_features[categorical_features]

print(feat_num.head(4).transpose())
print(feat_cat.head(4).transpose())

#5.2 Box cox transform for skewd numerical data
#Another transformation to reduce skew. 

# Plot skew value for each numerical value
skewness = feat_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False, inplace= True)

#Encode categorial features: can and should be replaced.

skewness = skewness[abs(skewness) > 0.5]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
print("Mean skewnees: {}".format(np.mean(skewness)))

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    feat_num[feat] = boxcox1p(feat_num[feat], boxcox_normmax(feat_num[feat] + 1))
    data_features[feat] = boxcox1p(data_features[feat], boxcox_normmax(data_features[feat] + 1))
    
skewness.sort_values(ascending=False)

#Observe the correction. We can see that a lot of parameters remained skewd. I suspect that's for variables that have a lot of 0.
skewness = feat_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features after Box Cox transform".format(skewness.shape[0]))
print("Mean skewnees: {}".format(np.mean(skewness)))
skewness.sort_values(ascending=False)



# 6. Adding features
# 6.1 Creating features from the data
#Adding features at this section to be able to view them at the visualization section next

# Calculating totals before droping less significant columns

#  Adding total sqfootage feature 
data_features['TotalSF']=data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']
#  Adding total bathrooms feature
data_features['Total_Bathrooms'] = (data_features['FullBath'] + (0.5 * data_features['HalfBath']) +
                               data_features['BsmtFullBath'] + (0.5 * data_features['BsmtHalfBath']))
#  Adding total porch sqfootage feature
data_features['Total_porch_sf'] = (data_features['OpenPorchSF'] + data_features['3SsnPorch'] +
                              data_features['EnclosedPorch'] + data_features['ScreenPorch'] +
                              data_features['WoodDeckSF'])


# data_features['Super_quality'] = OverallQual * 
# vars = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath']



#6.2 Deleting features
#Features that cant be skewd or are unsignificant.
data_features['haspool'] = data_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data_features['hasgarage'] = data_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data_features['hasbsmt'] = data_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data_features['hasfireplace'] = data_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# Not normaly distributed can not be normalised and has no central tendecy
col_to_drop = ['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF', 'PoolArea','3SsnPorch',
               'LowQualFinSF','MiscVal','BsmtHalfBath','ScreenPorch','ScreenPorch','KitchenAbvGr','BsmtFinSF2',
               'EnclosedPorch','LotFrontage','BsmtUnfSF','GarageYrBlt']

for col in col_to_drop:
    if col in data_features.columns:
        data_features.drop(col, axis=1, inplace=True)


print('data_features size:', data_features.shape)


#5.9 Splitting the data back to train and test
train = data_features.iloc[:len(y_train), :]
test = data_features.iloc[len(y_train):, :]
print('Train data shpe: ',train.shape)
print('Prediction on (Sales price) shape: ', y_train.shape)
print('Test shape: ', test.shape)



#7.Plotting the data
#7.1 Visually comparing data to sale prices
#One can observe the behaviour of the variables, locate outlier and more.

vars = data_features.columns
# vars = numerical_features
figures_per_time = 4
count = 0 
y = y_train
for var in vars:
    x = train[var]
#     print(y.shape,x.shape)
    plt.figure(count//figures_per_time,figsize=(20,4))
    plt.subplot(1,figures_per_time,np.mod(count,4)+1)
    plt.scatter(x, y);
    plt.title('f model: T= {}'.format(var))
    count+=1
    


# Removes outliers 
outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])
y_train = train['SalePrice']



#7.2 Comparing data to sale price through correlation matrix
#Numerical values correlation matrix, to locate dependencies between different variables.

# Complete numerical correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True)

#Only print high correlation
high_corr = 0.8
col_high_corr = []
for col in corrmat.columns:
    if sum(corrmat[col] > high_corr) > 1: #column with high correlation
        col_high_corr.append(col)

#Only print the correlation matrix for the high correlation
corrmat_high = corrmat.loc[col_high_corr,col_high_corr] > high_corr
sns.heatmap(corrmat_high[corrmat_high], vmax=1, square=True)



#7.3 Pairplot for the most intresting parameters
# pair plots for variables with largest correlation
# saleprice correlation matrix
#nlargest: Return the largest n elements.
corr_num = 15 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, col_Target)[col_Target].index #retourne les plus plus fortes corrélation
var_num = 8
vars = cols_corr[0:var_num]

sns.set()
sns.pairplot(train[vars], size = 1.5)
plt.show()



#8. Preparing the data
#Dropping Sale price, Creating dummy variable for the categorial variables and matching 
#dimentions between train and test
data_features = data_features.drop(col_Target, axis = 1)
final_features = pd.get_dummies(data_features)

print(final_features.shape)
X = final_features.iloc[:len(y), :]
X_test = final_features.iloc[len(y):, :]
X.shape, y_train.shape, X_test.shape


print(X.shape,y_train.shape,X_test.shape)

#Remove overfitting
# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.95:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')

X = X.drop(overfit, axis=1).copy()
X_test = X_test.drop(overfit, axis=1).copy()

print(X.shape,y_train.shape,X_test.shape)


#9. Creating the model
#9.1 Importing learning libraries
from datetime import datetime
#RobustScaler: Scale features using statistics that are robust to outliers.
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold #K-Folds cross-validator
from sklearn.model_selection import  cross_val_score #Evaluate a score by cross-validation
from sklearn.metrics import mean_squared_error #Mean squared error regression loss
#The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms and estimators.
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeCV #Ridge regression with built-in cross-validation.
from sklearn.linear_model import ElasticNetCV  #Model specific cross-validation, Tuning the hyper-parameters of an estimator
from sklearn.linear_model import LassoCV  #Linear model iterative fitting along a regularization path.
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR #Support Vector Machine
from mlxtend.regressor import StackingCVRegressor
from sklearn import linear_model
from xgboost import XGBRegressor #XGBoost stands for “Extreme Gradient Boosting”, supervised learning
from lightgbm import LGBMRegressor

#Méthode ensembliste
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosting for regression.
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor



#9.2 Defining folds and score functions
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=kfolds))
    return (rmse.mean())

# rmsle scoring function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


#9.3 Defining models
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

#make_pipeline permet d'ajouter un scaler pour définir le modèle
lightgbm = LGBMRegressor(objective='regression')
xgboost = XGBRegressor(max_depth=3)
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, cv=kfolds))                                
svr = make_pipeline(RobustScaler(), SVR(max_iter=1e7))
linr = linear_model.LinearRegression()
lasso = linear_model.Lasso(alpha=0.1)

#test of the model
cv_rmse(model=lightgbm)
cv_rmse(model=xgboost)
cv_rmse(model=ridge)
cv_rmse(model=lasso)
cv_rmse(model=elasticnet)
cv_rmse(model=svr)
cv_rmse(model=linr)

#Régression linéaire sur les modèles pour trouver les coefficients à appliquer
list_models=['lightgbm', 'xgboost', 'ridge', 'lasso', 'elasticnet', 'svr', 'linr']
train_predictions_models = pd.DataFrame(columns=list_models)
test_predictions_models = pd.DataFrame(columns=list_models)

lightgbm.fit(X,y)
train_predictions_models['lightgbm'] = lightgbm.predict(X)
test_predictions_models['lightgbm'] = lightgbm.predict(X_test)
xgboost.fit(X,y)
train_predictions_models['xgboost'] = xgboost.predict(X)
test_predictions_models['xgboost'] = xgboost.predict(X_test)
ridge.fit(X,y)
train_predictions_models['ridge'] = ridge.predict(X)
test_predictions_models['ridge'] = ridge.predict(X_test)
lasso.fit(X,y)
train_predictions_models['lasso'] = lasso.predict(X)
test_predictions_models['lasso'] = lasso.predict(X_test)
elasticnet.fit(X,y)
train_predictions_models['elasticnet'] = elasticnet.predict(X)
test_predictions_models['elasticnet'] = elasticnet.predict(X_test)
svr.fit(X,y)
train_predictions_models['svr'] = svr.predict(X)
test_predictions_models['svr'] = svr.predict(X_test)
linr.fit(X,y)
train_predictions_models['linr'] = linr.predict(X)
test_predictions_models['linr'] = linr.predict(X_test)
lasso.fit(X,y)
train_predictions_models['lasso'] = lasso.predict(X)
test_predictions_models['lasso'] = lasso.predict(X_test)


reg_of_models = linear_model.LinearRegression()
reg_of_models.fit(train_predictions_models,y)
df_coef = pd.DataFrame(reg_of_models.coef_, index=list_models, columns=['Coef'])
print(df_coef.round(3))

train_pred_combined = pd.DataFrame(np.zeros([len(train),1]), columns=['Train_Pred_combined'])
test_pred_combined = pd.DataFrame(np.zeros([len(test),1]), columns=['Test_Pred_combined'])
for mod in df_coef.index:
    train_pred_combined['Train_Pred_combined'] = train_pred_combined['Train_Pred_combined'] + train_predictions_models[mod] * df_coef.loc[mod,'Coef']
    test_pred_combined['Test_Pred_combined'] = test_pred_combined['Test_Pred_combined'] + test_predictions_models[mod] * df_coef.loc[mod,'Coef']



#Submission
submission = pd.DataFrame(columns=[col_Id, col_Target])
submission[col_Id] = test_ID
submission[col_Target] = np.expm1(test_pred_combined)

submission.to_csv('2020_1_16_house_Price_submissions.csv', header=True, index=False)





