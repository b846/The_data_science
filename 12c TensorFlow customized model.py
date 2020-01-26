#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:03:58 2019

@author: b
Réalisation du model
Inputs x, z
outputs y
Linear model
there is no depth
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
training_data = np.load('/home/b/Documents/Python/Data/TF_intro.npz')
input_size = 2
output_size = 1

#tf.keras.sequential() function that specifies how the model will be laid down ('stack layers')
#linear combinaison + output = layer
#outputs = np.dot(inputs,weights)+bias
#tf.keras.layers.Dense(output size): takes the inputs provided to the model and calculates the dot product
#tf.keras.layers.Dense(output size): of the inputs and the weights and adds the bias
#kernel_initializer: initialize the weights
#bias_initializer: initialize the bias
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                                  bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)) 
                            ])

#tf.keras.optimizers.SGD(learning_rate): stochastic gradient descent optimizers, including support for learning rate, momentum, decay, etc.
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


#model.compile(optimizer, loss): configures the model for training
#SGD = Stochatic Gradient Descent
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#ici, on a remplacé l'optimizer SGD par un optmizer personalisé
#3. Change the loss function. An alternative loss for regressions is the Huber loss. 
#The Huber loss is more appropriate than the L2-norm when we have outliers, as it is less sensitive to them (in our example we don't have outliers, but you will surely stumble upon a dataset with outliers in the future). The L2-norm loss puts all differences *to the square*, so outliers have a lot of influence on the outcome. 
#The proper syntax of the Huber loss is 'huber_loss'
model.compile(optimizer=custom_optimizer, loss='huber_loss')

#model.fit(inputs, targets) fits (trains) the model
#verbose = 0, stands for 'silent' or no outputs about the training is displayed
#verbose = 1, stands for 'progress bar'
model.fit(training_data['inputs'], training_data['targets'], epochs=20, verbose=1)


#Extract the weights and bias
weights = model.layers[0].get_weights()[0]
 
bias = model.layers[0].get_weights()[1]
print(bias)

#Extract the outputs (make predictions
model.predict_on_batch(training_data['inputs'])     #values based on the trained model
training_data['targets'].round(1)

#Plotting the data
#this line should be as close to 45 degrees as possible
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()