from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

def f(x,y):
    return np.sin(x)+np.sin(y)+np.sin(x+y)

plt.close()
ax = Axes3D(plt.figure())
x = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x, x)
Z = f(X,Y)
ax.plot_surface(X, Y, Z, color='lightblue')
plt.show()

#ligne de niveau issue du point (0.7,0.7) en rouge
def tg(u, t):
    v = np.cos(u[0]+u[1])
    return [-np.cos(u[1])-v, np.cos(u[0])+v]
    
t = np.linspace(0, 5., 200)

x0, y0 = 0.7, 0.7
sol = odeint(tg, [x0,y0], t)

xsol, ysol = sol[:,0], sol[:,1]
zsol = f(x0,y0)*np.ones(200)
plt.plot(xsol, ysol, zsol, color= 'r')
plt.show()

#ligne de plus grande pente issue du point (0.,0.)
def g(u, t):
    v = np.cos(u[0]+u[1])
    return [np.cos(u[0])+v, np.cos(u[1])+v]

x0, y0 = 0., 0.    
sol = odeint(g, [x0, y0], t)
xsol, ysol = sol[:,0], sol[:,1]
zsol = f(xsol, ysol)
plt.plot(xsol, ysol, zsol, color= 'b')
plt.show()
#approximation vraisemblable du maximum
print(xsol[-1], ysol[-1], zsol[-1])
#valeurs théoriques (cf exercice 763 corrigé par Baptiste Langlois)
print(np.pi/3, 3*3**0.5/2)