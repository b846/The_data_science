# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:27:28 2019

@author: Briac
"""

from pylab import *

def f(x):
    return sin(x) /(1+x**2)

#1er graphe
xs = np.linspace(0,8*pi,256,endpoint=True) #abscisse
plot(xs, f(xs))
show()

#2ème graphe
figure(figsize=(8,6)) #augmente la taille de la figure à 8cm * 6 cm
title("Courbe de y = f(x)")
xlabel("x")
ylabel("y")
grid(True) #ajout d'une grille
xlim([0,8*pi]) #limite le rendu à la plage [O, 8pi] en abscisse
plot(xs, f(xs))
show()

#3ème graphe
plot(xs, f(xs), linestyle ='--', linewidth=3, color ='red')
show()

#4ème graphe
xs2 = np.linspace(0, 8*pi,32)
plot(xs, f(xs), linestyle ='solid', marker='s')
show()

#Moyenne écart type
l = []
for i in range(0,300):
    l.append(f(5))
print (l)
mean(l)
sqrt(5)
std(l)  #écart type



