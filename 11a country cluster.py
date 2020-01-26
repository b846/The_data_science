#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:00:32 2019

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

#plot the data
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-100,180)
plt.ylim(-90,90)
plt.show

#select the features
x = data.iloc[:,1:3]    #slices the data fram, given rows and columns to be kept

#Clustering
kmeans = KMeans(3)      # KMeans method imported from sklearn, k is the number of clusters, kmeans is an object
kmeans.fit(x)
#returns the cluster predictions in an array
identified_clusters = kmeans.fit_predict(x)
identified_clusters# contain the predict cluster
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters

#plot the data
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-100,180)
plt.ylim(-90,90)
plt.show

