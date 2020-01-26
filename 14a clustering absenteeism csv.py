#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:15:20 2019

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

#load the data np
raw_data_csv_np = np.loadtxt('/home/b/Documents/Python/Data/Absenteeism-data.csv', delimiter = ',', dtype = 'str')
print(raw_data_csv_np.shape)
unscaled_inputs_all = raw_data_csv_np[:,1:-1]      # drop the ID column
targets_all = raw_data_csv_np[:,-1]    #last column

print(raw_data_csv_np[0,:])        # identifiants des colonnes
print(pd.unique(raw_data_csv_np[:,1]))     #Reasons for absenteism

#load the data pd
raw_data_csv_pd = pd.read_csv('/home/b/Documents/Python/Data/Absenteeism-data.csv')  
print(raw_data_csv_pd.head(6))
print(raw_data_csv_pd.describe(include='all'))

#plot of the data
plt.scatter(raw_data_csv_pd['Reason for Absence'],raw_data_csv_pd['Absenteeism Time in Hours'])
plt.title('Absenteeism Time in Hours = f(Reason for Absence)')
plt.xlabel('Reason for Absence')    
plt.ylabel('Absenteeism Time in Hours')
plt.show

#select the features
x = raw_data_csv_pd.copy()
x = raw_data_csv_pd['Reason for Absence']
y = raw_data_csv_pd['Absenteeism Time in Hours']
xy = pd.concat([x, y], axis = 1)

#Clustering
kmeans = KMeans(2)      # KMeans method imported from sklearn, k is the number of clusters, kmeans is an object
kmeans.fit(xy)           #returns the cluster predictions in an array

#Standardize the variables
from sklearn import preprocessing
xy_scaled = preprocessing.scale(xy)           #standardize the variables separatly

# selecting the number of clusters with the elbow method
kmeans.inertia_               #WCSS
wcss = []
for i in range(1,10):
    kmeans =KMeans(i)       #calcul du coefficient pour un nombre de cluster variant de 1 à 6
    kmeans.fit(xy)           #calcul input data
    wcss_iter = kmeans.inertia_        #calcul du WCSS with the inertia method
    wcss.append(wcss_iter)              #wcss is decreasing as the number of clusters increases
    
print(wcss)

#plot the evolution of wcss wuth the clusters
number_clusters = range(1,10)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')    
plt.ylabel('within-cluster sum of squares')
plt.show()

#Exploring clustering solutions and select the number of cluster
kmeans_new = KMeans(5)
kmeans_new.fit(xy_scaled)
clusters_new = xy.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(xy_scaled)
print(clusters_new)
#we plot the original values and the cluster_pred based on the standardized data
plt.scatter(clusters_new['Reason for Absence'],clusters_new['Absenteeism Time in Hours'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.title('bsenteeism Time in Hours = f(Reason for Absence) original data with the cluster based on the standardized data')
plt.xlabel('Reason for Absence')    
plt.ylabel('Absenteeism Time in Hours')
plt.show()


