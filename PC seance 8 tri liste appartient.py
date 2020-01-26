
# coding: utf-8

# #Séance 8 : 25 novembre 2015

# ##Sélection

# ###Détermination de la valeur maximale d'une liste (non vide) de nombres.

# In[1]:

def maximum(liste):
    """Retourne la valeur maximum dans une liste non vide de nombres."""
    maxi = liste[0]
    for valeur in liste:
        if valeur > maxi:
            maxi = valeur
    return maxi

L = [-3, 7, 6, 8, -5]
print(maximum(L))


# In[2]:

L.append(9)
print(maximum(L))


# ###Détermination de la position de maximum d'une liste (non vide) de nombres.

# In[3]:

def indice_max(liste):
    """Retourne le plus petit indice où se trouve le maximum dans une liste non vide de nombres."""
    imaxi = 0
    n = len(liste)
    for i in range(n):
        if liste[i] > liste[imaxi]:
            imaxi = i
    return imaxi

print(indice_max(L))


# In[4]:

print(L)


# In[5]:

L.pop()
print(L)


# In[6]:

print(indice_max(L))


# ###Tri par sélection

# <li>détermination d'une position du maximum
# <li>échange pour amener le maximum en dernière position
# <li>répéter avec une donnée en moins, jusqu'à ce que toutes les toutes les valeurs soient rangées dans l'ordre

# In[7]:

def i_max(liste, p):
    """Retourne le plus petit indice où se trouve le maximum dans liste[:p] de nombres."""
    imaxi = 0
    for i in range(p):
        if liste[i] > liste[imaxi]:
            imaxi = i
    return imaxi

def echange(liste, i, j):
    """Echange liste[i] et liste[j]."""
    liste[i], liste[j] = liste[j], liste[i]
    return None

def tri_selection(liste):
    n = len(liste)
    for p in range(n, 0, -1): #p=longueur restant à trier
        i = i_max(liste, p)
        echange(liste, i, p-1)
    return None

tri_selection(L)
print(L)        


# ###Complexité

# Notons $C(n)$ le nombre d'opérations pour le tri par sélection d'une liste de taille $n$ <b> dans le pire des cas</b>, et de même $I(n), E(n)$ la complexité des fonctions <tt>i_max</tt> et <tt>echange</tt>.
# <li>$E(n)=3$ (indépendamment de $n$, puisqu'il y a accès direct aux "cases" du tableau)
# <li>$I(n)=1+3n$ ($n$ affectations du compteur $i$, $n$ tests et au plus $n$ affectations de <tt>imaxi</tt>)
# <li>$\displaystyle C(n)=1+\sum_{p=1}^n{(1+I(p)+2+3)}=1+\sum_{p=1}^n{(3p+7)}=\frac{3}{2}n^2+\frac{17}{2}n+1\sim \frac{3}{2}n^2$ quand $n\to +\infty$.
# <p>La complexité du tri par sélection est <b>quadratique</b>.

# Variante avec le rangement des petits au début (ce qui permet de n'écrire que des incrémentations de compteur) :

# In[8]:

def i_min(liste, p):
    """Retourne le plus petit indice où se trouve le minimum dans liste[p:] de nombres."""
    n = len(liste)
    imini = p
    for i in range(p, n):
        if liste[i] < liste[imini]:
            imini = i
    return imini

def tri_selection(liste):
    n = len(liste)
    for p in range(n): #p=nombre de petits rangés au début
        i = i_min(liste, p)
        echange(liste, i, p)
    return None

L = [-3, 7, 6, 8, -5]
tri_selection(L)
print(L)


# ## Insertion

# ###Recherche d'un élément (et de sa position éventuelle) dans une liste croissante

# C'est mal parti...il faudra mettre au point un invariant de boucle maintenu par l'itération (<tt>while</tt>)

# In[ ]:

def appartient(liste, e):
    """Retourne (False, None) si e n'est pas présent dans la liste, (True,indice) si e==liste[i].
    Recherche dichotomique."""
    fin = len(liste)-1
    if e<liste[0] or e>liste[fin]:
        return (False, None)
    else:
        debut = 0
        while debut != fin:
            milieu = (fin + debut)//2
            if e<=liste[milieu]:
                fin = milieu
            else:
                debut = milieu
        return 
    


# In[ ]:



