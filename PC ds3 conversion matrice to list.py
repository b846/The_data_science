
# coding: utf-8

# #DS3 algorithmes sur les graphes

# ##Partie 1

# ###Saisie des exemples

# In[1]:

exemple1 = [[1,2,3,4,5,6], [0,2], [0,1], [0,4], [0,3], [0,6], [0,5]]
exemple2 = [[1,2,3,4], [0,2], [0,1], [0], [0]]
exemple3 = [[1,3], [0,2], [1,3], [0,2]]


# ##Partie 2

# In[4]:

import numpy as np

A = np.array([[0,1,1,1,1], [1,0,1,0,0], [1,1,0,0,0], [1,0,0,0,0], [1,0,0,0,0]], dtype = np.uint8)
print(A)


# In[5]:

sommets1 = [i for i in range(len(exemple1))] 
print(sommets1)


# In[7]:

[i for i, v in enumerate(exemple1)] #autre possibilité


# In[14]:

def listeToMatrice(V, n):
    a = np.zeros([n,n], dtype=np.uint8)
    for i in range(n):
        for j in V[i]:
            a[i, j] = 1
    return a

A1 = listeToMatrice(exemple1, 7)
print(A1)


# In[15]:

def matriceToListe(M, n):
    V = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i, j] == 1:
                V[i].append(j)
    return V

V2 = matriceToListe(A, 5)
print(V2)
V1 = matriceToListe(A1, 7)
print(V1)


# In[17]:

def nbVoisinsCommuns(A, n, i, j):
    commun = 0 #1 affectation
    for k in range(n): #n passages dans la boucle
        if A[i,k] == 1 and A[j,k] == 1: #2 comparaisons
           commun = commun + 1 #sous condition, 1 +, 1 affectation
    return commun


# ###Complexité
# <p> $4n+1$ dans le pire des cas

# In[23]:

def testC(M, n):
    for i in range(n): #n passages dans le pire des cas c'est à dire si la propriété est vérifiée (return False peut interrompre avant)
        for j in range(i): #i passages (idem)
            if nbVoisinsCommuns(M, n, i, j) != 1: #complexité 1+4n+1
                return False
    return True
    
print(testC(A1, 7))
print(testC(A, 5))
print(testC(listeToMatrice(exemple3, 4), 4))


# ###Complexité
# <p>$\displaystyle \sum_{i=0}^{n-1}{\sum_{j= 0}^{i-1}{(4n+2)}}=(4n+2)\sum_{i=0}^{n-1}{i}=(4n+2)\frac{n(n-1)}{2}\sim 2n^3$ quand $n\to+\infty$ (ce n'est pas fameux).

# In[21]:

def testH(V, n):
    for i in range(n):
        if len(V[i]) == n-1:
            return True
    return False

print(testH(exemple1, 7))
print(testH(exemple2, 5))
print(testH(exemple3, 4))


# ##Partie 3

# In[26]:

import  scipy.linalg as lg

print(lg.eigvals(A1))


# ##Partie 6

# In[28]:

exemple4 = [[1,4], [0,2], [1,3], [2,4], [0,3]]


# In[34]:

def graphe(p):
    V = [[1, 4], [0, 2], [1, 3], [2, 4], [0, 3]]
    n = 5
    for k in range(p):
        N = n
        A = listeToMatrice(V, N)
        for i in range(N):
            for j in range(i):
                if nbVoisinsCommuns(A, N, i, j)==0:
                    V[i].append(n)
                    V[j].append(n)
                    V.append([i,j])
                    n = n+1
    return V

print(graphe(1))
print(graphe(2))
print(len(graphe(3)))


# In[ ]:



