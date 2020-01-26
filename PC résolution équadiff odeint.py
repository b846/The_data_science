
# coding: utf-8

# #Calcul approché de solution d'équation différentielle linéaire

# ##Une seule équation scalaire $x'(t)=a(t)x(t)+b(t)$

# In[16]:

##Calcul d'une valeur
def euler1(a, b, t0, x0, t_fin, n):
    """Retourne la valeur approchée de la solution de l'ode x'(t)=a(t)*x(t)+b(t) à l'instant t 
    avec la condition initiale x(t0)=x0 et n pas intermédiares."""
    pas = (t_fin-t0)/n
    x, t = x0, t0
    for k in range(n):
        x, t = x + pas * (a(t) * x + b(t)), t + pas
    return (x, t)


# In[17]:

from math import exp, sin

print(euler1(exp, sin, 0, 0, 1, 10))
print(euler1(exp, sin, 0, 0, 1, 100))
print(euler1(exp, sin, 0, 0, 1, 1000))
print(euler1(exp, sin, 0, 0, 1, 10000))


# In[18]:

#comparaison avec scipy
def f(x, t): #déclaration de la fonction
    return exp(t)*x+sin(t)


# In[19]:

from scipy.integrate import odeint

print(odeint(f, 0, [0, 1]))


# ##Etude du système d'équations $x'_1(x)=-x_2(t)+1\\x'_2(t)+x_1(t)-2x_2(t)+t$

# ###1) à partir de l'expression calculée par réduction : $x_1(t)=(-3-2t)\exp(-t)-t+4\\x_2(t)=(-1-2t)\exp(-t)+2$

# In[20]:

##Représentation graphique
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[21]:

t= np.linspace(0, 2, 100)
x1 = (-3-2*t)*np.exp(-t)-t+4
x2 = (-1-2*t)*np.exp(-t)+2
plt.plot(x1, x2, color='k')


# ###2) par la méthode d'Euler

# In[22]:

def euler(t0, x0, y0, A, B, t_fin, n): #estimation de la solution à l'instant t_fin
    """t0,x0,y0:CI, A=2-liste de 2-listes, B=2-liste, t_fin=instant visé, n=nb de pas."""
    pas = (t_fin-t0)/n
    t, x, y = t0, x0, y0
    for k in range(n):
        a, b = A(t), B(t) 
        t, x, y = t+pas, x+pas*(a[0][0]*x+a[0][1]*y+b[0]), y+pas*(a[1][0]*x+a[1][1]*y+b[1])
    return (t, x, y) 


# In[23]:

def A(t):
    return [[0, -1], [1, -2]]

def B(t):
    return [1, t]

print(euler(0, 1, 1, A, B, 2, 10))


# Noter le cumul des arrondis qui empêche d'atteindre exactement <tt>t_fin=2</tt>. Idem ci-dessous.

# In[24]:

print(euler(0, 1, 1, A, B, 2, 100))


# On peut aussi mémoriser les résultats intermédiaires en vue de tracer la courbe calculée.

# In[25]:

def euler_liste(t0, x0, y0, A, B, t_fin, n):
    t, x, y = t0, x0, y0
    lt, lx, ly = [t], [x], [y]
    pas = (t_fin-t0)/n
    for k in range(n):
        a, b = A(t), B(t)
        t, x, y = t+pas, x+pas*(a[0][0]*x+a[0][1]*y+b[0]), y+pas*(a[1][0]*x+a[1][1]*y+b[1])
        lt.append(t); lx.append(x); ly.append(y)
    return (lt, lx, ly) 


# In[26]:

t, x, y = euler_liste(0, 1, 1, A, B, 2, 100)
plt.plot(x, y)


# ###3) avec une fonction de <tt>scipy</tt>

# In[27]:

from scipy.integrate import odeint

def f(x,t): #x=2-vecteur
    return [-x[1]+1, x[0]-2*x[1]+t]

t = np.linspace(0, 2, 100)
r = odeint(f, [1,1], t)
print(r.shape)


# Il faut séparer les deux <b>colonnes</b> d'où le slicing :

# In[28]:

plt.plot(r[:,0], r[:,1])


# ###4) comparaison graphique

# In[29]:

t= np.linspace(0, 2, 100)
x1 = (-3-2*t)*np.exp(-t)-t+4
x2 = (-1-2*t)*np.exp(-t)+2
plt.plot(x1, x2, color='k')

r = odeint(f, [1, 1], t)
x, y = r[:,0], r[:,1]
plt.plot(x, y, ls=' ', marker='+', color='k')

t, x, y = euler_liste(0, 1, 1, A, B, 2, 10)
plt.plot(x, y, ls=' ', color='r', marker='o')

t, x, y = euler_liste(0, 1, 1, A, B, 2, 100)
plt.plot(x, y, ls=' ', marker='o', color='b')


# ###5) adaptation à des coefficients non constants $x'_1(t)=tx_1(t)-x_(t)+1\\x'_2(t)=x_1(t)-2x_2(t)+t$

# In[30]:

def A(t): #pour euler_liste
    return[[t, -1], [1, -2]]

t, x, y = euler_liste(0, 1, 1, A, B, 2, 10)
plt.plot(x, y, ls=' ', color='r', marker='o')

def f(x, t): #pour odeint
    return [t*x[0]-x[1]+1, x[0]-2*x[1]+t]

t = np.linspace(0, 2, 100)
r = odeint(f, [1, 1], t)
x, y = r[:,0], r[:,1]
plt.plot(x, y, ls=' ', marker='+', color='k')


# In[ ]:



