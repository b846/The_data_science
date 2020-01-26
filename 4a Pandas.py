# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:52:28 2019

@author: Briac
 DataFrame.at : Access a single value for a row/column label pair.
 |      DataFrame.loc : Access a group of rows and columns by label(s).
 |      DataFrame.iloc : Access a group of rows and columns by integer position(s).
"""

###"Serie : objet étiqueté de type tableau unidimensionnel 
#capable de contenir n'importe quel type d'objet ;
###DataFrame : structure de données étiquetée en deux dimensions 
#dans laquelle les colonnes peuvent être de types différents ;
##"Panel : une structure de données en trois dimensions. 
#Vous pouvez les considérer comme des dictionnaires de DataFrames.

import pandas as pd
import numpy as np
series = pd.Series([1,2,3,4,5, np.nan, "a string", 6])
df = pd.DataFrame(np.array([1,2,3,4,5,6]).reshape(2,3))
df = pd.DataFrame(np.array([1,2,3,4,5,6]).reshape(2,3), columns=list('ABC'), index=list('XY'))
series = pd.Series([1,2,3,4,5, np.nan, "a string", 6])


series2 = pd.Series([1,2,np.nan, 4])
df = pd.DataFrame(np.array([1,2,3,4,5,6]).reshape(2,3))
df = pd.DataFrame(np.array([1,2,3,4,5,6]).reshape(2,3), columns=list('ABC'), index=list('XY'))

#2ème set de data
df2 = pd.DataFrame(np.arange(1, 7501).reshape(500,15))
df2.head(2)   # montre les premières lignes
df2.tail()    # montre les dernières lignes

# Description des données
df3 = pd.DataFrame(np.arange(1, 100, 0.12).reshape(33,25))
df3.describe()

# Découpage en frange de données
df3 = pd.DataFrame(np.arange(1, 100, 0.12).reshape(33,25))
df3.iloc[:5,:10]  #Sélection des 5 premières lignes et des 10 premières colonnes

# Renommage des colonnes selon A, B, C, D...
# Utilisatin de la fonction lambda
# c est l'entête actuel de l'axe, ici les colonnes
df4 = df3.rename(columns=lambda c: chr(65+c))


# Lecture de fichier SCV
chemin = 'C:/Users/Briac/Documents/Data/'
baby_names = pd.read_csv(chemin + 'Popular_Baby_Names.csv')

# Trie des valeurs
baby_names.sort_values(by='Count').head()
baby_names.sort_values(by='Count', ascending=False).head()

# Indices
a = pd.DataFrame(np.arange(10).reshape(5,2),columns=['c1','c2'])
print(a)
a.ix[0,1] 
a.c2[0] 
a.c2.ix[0]
a.c1[a.c1 == 8].index.tolist()
a.loc[a['c1'] == 8].index[0]