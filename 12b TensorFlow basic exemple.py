#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 07:01:29 2019

@author: b
Réalisation du model
Inputs x, z
outputs y
Linear model
there is no depth
"""

#import the relevant librairies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

# load the data
observations = 1000
xs=np.random.uniform(low=-10, high=10, size=(observations,1))   #number of observations by number of observations
zs=np.random.uniform(low=-10, high=10, size=(observations,1))
generated_inputs = np.column_stack((xs,zs))
print('generated_inputs.shape:', generated_inputs.shape)

#Create the target we will aim at
noise = np.random.uniform(-1,1,size=(observations,1))
generated_targets = 2*xs - 3*zs + 5 + noise
print('generated_targets.shape:', generated_targets.shape)

#save n-dimensional arrays in .npz format using a certain keyword (label) for each array
#here the label is "inputs" and the array is "generated_inputs"
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)


# Solving with TensorFlow
training_data = np.load('/home/b/Documents/Python/Data/TF_intro.npz')
input_size = 2
output_size = 1

#tf.keras.sequential() function that specifies how the model will be laid down ('stack layers')
#linear combinaison + output = layer
#outputs = np.dot(inputs,weights)+bias
#tf.keras.layers.Dense(output size): takes the inputs provided to the model and calculates the dot product
#tf.keras.layers.Dense(output size): of the inputs and the weights and adds the bias
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size)
                            ])

#model.compile(optimizer, loss): configures the model for training
#SGD = Stochatic Gradient Descent
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
model.compile(optimizer='sgd', loss='mean_squared_error')

#model.fit(inputs, targets) fits (trains) the model
#verbose = 0, stands for 'silent' or no outputs about the training is displayed
#verbose = 1, stands for 'progress bar'
model.fit(training_data['inputs'], training_data['targets'], epochs=20, verbose=0)


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

