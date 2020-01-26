#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:49:23 2019

@author: b
Réalisation du model
Inputs x, z
outputs y
Linear model
there is no depth
action: the graphs plotted show give a give overview of the pb
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
import numpy as np                                      #mathematical operations
import matplotlib.pyplot as plt                        #nice graphs
from mpl_toolkits.mplot3d import Axes3D              #nice 3d graph

# load the data
observations = 1000
xs=np.random.uniform(low=-10, high=10, size=(observations,1))   #number of observations by number of observations
zs=np.random.uniform(low=-10, high=10, size=(observations,1))
inputs = np.column_stack((xs,zs))
print('inputs.shape: ', inputs.shape)

#Create the target we will aim at
noise = np.random.uniform(-3,3,size=(observations,1))
targets = 13*xs - 7*zs - 12 + noise
print('targets.shape: ', targets.shape)

#plot the training data
targets = targets.reshape(observations,)
my_targets = targets.reshape(observations,1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100)
plt.title('Targets')
plt.show()


def Neural_network(targets, nb_iteration=10, learning_rate=0.01, init_range = 1):
    #train the model, for each iterations: calculate outputs, 
    #compare outputs to targets through the loss
    weights = np.random.uniform(low=-init_range, high=init_range, size=(2,1))
    biases = np.random.uniform(low=-init_range, high=init_range, size=1)
    print('initial weights: ', weights)
    print('initial biases', biases)
    
    list_loss = []
    for i in range(nb_iteration):
        outputs = np.dot(inputs,weights) + biases
        deltas = outputs - targets
        
        loss = np.sum(abs(deltas)) / observations     #calcule de la loss function
        list_loss.append(loss)
        
        #update of the variables
        deltas_scaled = deltas / observations
        weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
        biases = biases - learning_rate * np.sum(deltas_scaled)
    return outputs, list_loss, loss, weights, biases


outputs, list_loss, loss, weights, biases = Neural_network(my_targets)

#plot of the evolution of the loss
plt.plot(list_loss)
plt.xlabel('nb of iterations')
plt.ylabel('loss')
plt.show()

#plot the trained data
plt.plot(outputs,targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()