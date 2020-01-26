#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:30:20 2019

@author: b
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Téléchargements')

#importation des bibliotèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# load the data
data = pd.read_csv('3.01. Country clusters.csv')  
data.head(6)
data.describe(include='all')

#map the data
data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})
data_mapped


#select the features
x = data_mapped.iloc[:,3:4]    #slices the data fram, given rows and columns to be kept

#Clustering
kmeans = KMeans(2)      # KMeans method imported from sklearn, k is the number of clusters, kmeans is an object
kmeans.fit(x)
#returns the cluster predictions in an array
identified_clusters = kmeans.fit_predict(x)
identified_clusters# contain the predict cluster
data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters

#plot the data
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show

# selecting the number of clusters
kmeans.inertia_               #WCSS
wcss = []
for i in range(1,7):
    kmeans =KMeans(i)       #calcul du coefficient pour un nombre de cluster variant de 1 à 6
    kmeans.fit(x)           #calcul input data
    wcss_iter = kmeans.inertia_        #calcul du WCSS with the inertia method
    wcss.append(wcss_iter)              #wcss is decreasing as the number of clusters increases
    
#plot the evolution of wcss wuth the clusters
number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')    
plt.ylabel('within-cluster sum of squares')
#le nombre de cluster à choisir est celui qui minimize le wcss, ici 2 ou 3
