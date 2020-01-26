# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:33:56 2019

@author: Briac
"""

# Compte le nombre de diviseurs stricts d'un nombre naturel non nul
# Pre  : n est un entier strictement positif
# Post : la valeur renvoyée contient le nombre de diviseurs stricts
#        de n
def nbdivisors(n):
    result = 0
    for i in range(1, n):
        if n % i == 0:
            result += 1
    return result

print('Le nombre 42 possède', nbdivisors(42), 'diviseurs stricts.')

# Divisors renvoie la liste des diviseurs de n
def divisors(n):
    result = []
    for i in range(1, n+1):
        if n % i == 0:
            result.append(i)
    return result

def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)

def pgcd(a, b):
    if b == 0:
        return a
    if a > b and b != 0:
        return pgcd(b, a % b)
    return pgcd(b, a)

def fibo(n):
    if n == 1 or n == 2:
        return 1
    return fibo(n-1) + fibo(n-2)

# somme des éléments
def sumall(data):
    result = 0
    for elem in data:
        result += elem
    return result

#valeur minimale
def findmin(data):
    result = data[0]
    for elem in data:
        if elem < result:
            result = elem
    return result

#Recherche d'une sous séquence
def issubsequence(subseq, seq):
    n = len(subseq)
    for i in range(0,len(seq)-n+1):
        if seq[i:i+n] == subseq:
            return True
    return False

#Nombre de voyelles
def nbvowels(s):
    result = 0
    for c in s:
        if c in 'aeiou':
            result += 1
    return result

#Caractères uniques
def uniquechars(s):
    seen = []
    for c in s:
        if c not in seen:
            seen.append(c)
    return seen

#Filtre
def filter_positive(data):
    result = []
    for elem in data:
        if elem > 0:
            result.append(elem)
    return result