# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:47:42 2019

@author: Briac
"""

#Installation de Numpy
#pip install numpy

values = [1, 2.4, 234, 112, 345]
import numpy as np
array = np.array(values)
array.dtype
array.shape

np.arange(25)
np.arange(25).reshape(5,5)
np.arange(10, 20, 0.4)
np.ones((3,3,3))
np.ones((3,3,3), dtype=np.int64)
empty_array = np.empty((3,6))

array = np.arange(10, 40, 0.4).reshape(5,5,3)
array[:, 2, 2]

#Filtre avec des booléns
#Permet de filtrer un tableau avec des booléens
a = np.arange(10, 40).reshape(5,6)
idx = np.array([[True, False, True, False, True, False],
                   [True, False, True, False, True, False],
                   [True, False, True, False, True, False],
                   [True, False, True, False, True, False],
                   [True, False, True, False, True, False]
                 ], dtype=bool)
a[idx]

#Autre façon
idx = a > 25



# Lecture de valeurs dans une matrice
m = np.array ([[0, 1, 2,3],[10,11,12,13],[20,21,22,23]])
np.delete(array,2, axis=0) #Détruit la 2èùme colonne de m
b = np.ones((3,6))



#Tableau de fonctions
def f(i,j):
    return 10*i +j

np.fromfunction(f,(4,5),dtype =int)


# Tirage pseudo-aléatoire
np.randon.rand(6)   # tableaux aléatoires de 6 nombres

