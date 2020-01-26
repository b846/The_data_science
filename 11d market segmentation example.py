#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:43:13 2019

@author: b
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# load the data
data = pd.read_csv('3.12. Example.csv')  
data.head(6)
data.describe(include='all')

plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.title('Loyalty = f(Satisfaction)')
plt.xlabel('Satisfaction')    
plt.ylabel('Loyalty')
plt.show
#we ca see two elliptical cluster
#we can divid the graph in 4 groups, low satisfacion high loyalty, low satisfaction, low loyalty, ...

#select the features
x = data.copy()

#Clustering
kmeans = KMeans(2)      # KMeans method imported from sklearn, k is the number of clusters, kmeans is an object
kmeans.fit(x)           #returns the cluster predictions in an array

#Clusters results
clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)

plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.title('Loyalty = f(Satisfaction)')
plt.xlabel('Satisfaction')    
plt.ylabel('Loyalty')
plt.show()
#the cluster seems to only take into account the satisfaction, because we have not standardize the data

#Standardize the variables
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)           #standardize the variables separatly
 
# selecting the number of clusters with the elbow method
kmeans.inertia_               #WCSS
wcss = []
for i in range(1,10):
    kmeans =KMeans(i)       #calcul du coefficient pour un nombre de cluster variant de 1 à 6
    kmeans.fit(x)           #calcul input data
    wcss_iter = kmeans.inertia_        #calcul du WCSS with the inertia method
    wcss.append(wcss_iter)              #wcss is decreasing as the number of clusters increases
    
wcss

#plot the evolution of wcss wuth the clusters
number_clusters = range(1,10)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')    
plt.ylabel('within-cluster sum of squares')
plt.show()
#Exploring clustering solutions and select the number of cluster
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
clusters_new
#we plot the original values and the cluster_pred based on the standardized data
plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.title('Loyalty = f(Satisfaction) original data with the cluster based on the standardized data')
plt.xlabel('Satisfaction')    
plt.ylabel('Loyalty')
plt.show()
#we need to name the cluster : alienated, supporters, fans, roamers

