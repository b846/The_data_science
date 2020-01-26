# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:04:46 2019

@author: Briac
"""

print('Hello World!')
height = 178
print('Sa taille est :')
print(height)

############" Chaine de caractère
address = "Promenade de l'Alma 50\n1200 Woluwé-Saint-Lambert"
print(address)

# \\Backslash
# \'Guillemet simple (apostrophe)
# \"Guillemet double
# \nSaut de ligne
# \rRetour chariot
# \tTabulation horizontale

type(123)
type('123')

year = 2016
birthyear = 1961
age = year - birthyear
print('Né en', birthyear, "j'ai", age, 'ans.')

day = 4
month = 8
year = 1961
print('Né le :', end=' ')
print(day, month, year, sep='/')

print('Né en ' + str(birthyear) + " j'ai " + str(age) + ' ans.')

firstname = input('Quel est ton prénom ? ')
print('Bonjour', firstname, 'et bienvenue !')


######### Type booléen
v = True
print(v)
print(type(v))

######### Exécution alternative
grade = 9.5
if grade >= 10:
    print("vous avez réussi")
else:
    print("vous avez raté")
    
n = 10
while n <= 5:
    print(n)
    n += 1
else:
    print("La boucle est terminée")
    
    
# Interruption de boucle
n = 1
while n <= 1000000:
    if n % 38 == 0 and n % 46 == 0:
        break
    n += 1
print(n, "est le plus petit nombre divisible par 38 et 46")
    
# Break permet de sorir de la boucle While
# Continue permet de revenir directement à l'exécution de la cdt while