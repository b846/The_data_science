#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 00:26:42 2020

@author: b
https://openclassrooms.com/fr/courses/4525266-decrivez-et-nettoyez-votre-jeu-de-donnees/4928126-tp-nettoyez-votre-jeu-de-donnees
"""

import pandas as pd # On importe la librairie Pandas, que l'on surnomme 'pd'

def lower_case(value): 
    print('Voici la valeur que je traite:', value)
    return value.lower()

data = pd.DataFrame([['A',1],
                     ['B',2],
                     ['C',3]], columns = ['lettre','position'])

nouvelle_colonne = data['lettre'].apply(lower_case)
nouvelle_colonne = nouvelle_colonne.values
print(nouvelle_colonne)
data['lettre'] = nouvelle_colonne
print(data)

# import des librairies dont nous aurons besoin
import pandas as pd
import numpy as np
import re

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

# chargement et affichage des données
data = pd.read_csv('personnes.csv')
print(data)

# Traitement des pays
VALID_COUNTRIES = ['France', 'Côte d\'ivoire', 'Madagascar', 'Bénin', 'Allemagne'
                  , 'USA']
                  
def check_country(country):
    if country not in VALID_COUNTRIES:
        print(' - "{}" n\'est pas un pays valide, nous le supprimons.' \
            .format(country))
        return np.NaN
    return country


# Traiter les emails
#Le problème avec cette colonne, c'est qu'il y a parfois 2 adresses email par ligne.
# Nous ne souhaitons prendre que la première. Nous créons donc la fonction  first  :
def first(string):
    parts = string.split(',')
    first_part = parts[0]
    if len(parts) >= 2:
        print(' - Il y a plusieurs parties dans "{}", ne gardons que {}.'\
            .format(parts,first_part))  
    return first_part


# Traiter les tailles
    #Nous aurons ici 2 fonctions :  convert_height, qui convertira les chaînes de caractères de type  
    #"1,34 m"  en nombre décimal, ainsi que  fill_height, qui remplacera les valeurs manquantes par la 
    #moyenne des tailles de l'échantillon.
def convert_height(height):
    found = re.search('\d\.\d{2}m', height)
    if found is None:
        print('{} n\'est pas au bon format. Il sera ignoré.'.format(height))
        return np.NaN
    else:
        value = height[:-1] # on enlève le dernier caractère, qui est 'm'
        return float(value)

def fill_height(height, replacement):
    if pd.isnull(height):
        print('Imputation par la moyenne : {}'.format(replacement))
        return replacement
    return height



# Appliquons toutes ces fonctions
data['email'] = data['email'].apply(first)
data['pays'] = data['pays'].apply(check_country)
data['taille'] = [convert_height(t) for t in data['taille']]
data['taille'] = [t if t<3 else np.NaN for t in data['taille']]
mean_height = data['taille'].mean()
data['taille'] = [fill_height(t, mean_height) for t in data['taille']]
data['date_naissance'] = pd.to_datetime(data['date_naissance'], 
                                           format='%d/%m/%Y', errors='coerce')
print(data)



# Aller plus loin: les compréhesions de listes
#ici, on applique une fonction à une liste
data = pd.read_csv('personnes.csv')

nouvelle_colonne = []
for t in data['taille']:
    nouvelle_colonne.append(convert_height(t))
data['taille'] = nouvelle_colonne

# on peut aussi utiliser apply
data = pd.read_csv('personnes.csv')

data['taille'] = data['taille'].apply(convert_height)


