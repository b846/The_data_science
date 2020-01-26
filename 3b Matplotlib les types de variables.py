#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:50:05 2020

@author: b
"""

import matplotlib.pyplot as plt

# VARIABLE QUALITATIVE
# Diagramme en secteurs
df_train['BsmtCond'].value_counts(normalize=True).plot(kind='pie')
# Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.axis('equal') 
plt.show() # Affiche le graphique

# Diagramme en tuyaux d'orgues
df_train['BsmtCond'].value_counts(normalize=True).plot(kind='bar')
plt.show()


# VARIABLE QUANTITATIVE
# Diagramme en bâtons
x_quantitative_continue.value_counts(normalize=True).plot(kind='bar')
plt.show()

# Histogramme
df_train['1stFlrSF'].hist(density=True)
plt.show()
# Histogramme plus beau
data[data.montant.abs() < 100]["montant"].hist(density=True,bins=20)
plt.show()

x_qualitative_ordinale = df_train['BsmtCond']
x_quantitative_discrete = df_train['1stFlrSF']
x_quantitative_continue = df_train['BsmtFinSF1']


