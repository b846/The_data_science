##copié à l'adresse https://openclassrooms.com/courses/la-programmation-scientifique-avec-python/scipy-une-trousse-a-outils-qu-elle-est-bien

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

# fréquence d’échantillonnage en Hz
fe = 100
# durée en secondes
T = 10
# Nombre de points :
N = T*fe
# Array temporel :
t = np.arange(1.,N)/fe
# fréquence du signal : Hz
f0 = 0.5
# signal temporel
sinus = np.sin(2*np.pi*f0*t)
# ajout de bruit
bruit = np.random.normal(0,0.5,N-1)
sinus2 = sinus + bruit
# signal fréquentiel : on divise par la taille du vecteur pour normaliser la fft
fourier = fftpack.fft(sinus2)/np.size(sinus2)
# axe fréquentiel:
axe_f = np.arange(0.,N-1)*fe/N
# On plot
plt.figure()
plt.subplot(121)
plt.plot(t,sinus2,'-')
plt.plot(t,sinus,'r-')
plt.xlabel('axe temporel, en seconde')
plt.subplot(122)
plt.plot(axe_f,np.abs(fourier),'x-')
plt.xlabel('axe frequentiels en Hertz')
plt.show()
