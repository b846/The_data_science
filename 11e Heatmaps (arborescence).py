#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:24:32 2019

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

# load the data
data = pd.read_csv('Country clusters standardized.csv', index_col='Country')  
data.head(6)
data.describe(include='all')
x_scaled = data.copy()
x_scaled = x_scaled.drop(['Language'], axis=1)
x_scaled


#Plot the data
sns.clustermap(x_scaled, cmap='mako')



