# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 08:39:09 2019

@author: Briac
"""

# Les séquences possibles sont
# Les listes : liste modifiable
address = ["Rue de la Loi", 16, 1000, "Bruxelles", "Belgique"]
numbers = [1, 2, 3, 4, 5]
print(numbers[0])
print(numbers[:3])
print(numbers[3:])
del(numbers[1:4])  # Suppression d'élements
numbers[0:0] = [0]  # Insertion d'éléments
essai = numbers + numbers # concaténation
print(essai)
a = [1, 2] * 4 # répétition
print(a)

i = -1
while i >= -len(numbers):
    print(numbers[i])
    i -= 1

# Appartenance dans une liste
def contains(data, element):
    i = 0
    while i < len(data):
        if data[i] == element:
            return True
        i += 1
    return False

print(contains(a, 4))
print(4 in a)

print(not contains(a, 2))
print(2 not in a)


# Les tuples : liste non modifiable d'éléments
a = ()  # Tuple vide
# Une fonction peut renvoyer 2 éléments
t = 1, 2, 3       # emballage
a, b, c = t       # déballage


# Les autres types de séquences
# - Chaine de caractère : séquence non modifiable de caractères
print("pa" in "papa")
s = "pa" * 2
p = s + " est là."


# - Intervalle
i = range(1, 5)
# Les piles