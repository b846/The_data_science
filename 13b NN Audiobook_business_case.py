#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:26:51 2019

@author: b
model :
inputs layer: 10
Nb hidden layer 2
hidden layer : 50
output layer: 2
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#import the relevant library
import numpy as np
import tensorflow as tf

#load the data
npz = np.load('/home/b/Documents/Python/Data/Audiobooks_data_train.npz')
train_inputs = npz['inputs'].astype(np.float)   #type float
train_targets = npz['targets'].astype(np.float)
print()

npz = np.load('/home/b/Documents/Python/Data/Audiobooks_data_validation.npz')
validation_inputs = npz['inputs'].astype(np.float)          #type float
validation_targets = npz['targets'].astype(np.float)

npz = np.load('/home/b/Documents/Python/Data/Audiobooks_data_test.npz')
test_inputs = npz['inputs'].astype(np.float)   #type float
test_targets = npz['targets'].astype(np.float)


#MODEL
#tf.keras.sequential() function that specifies how the model will be laid down ('stack layers')
#linear combinaison + output = layer
#outputs = np.dot(inputs,weights)+bias
#tf.keras.layers.Flatten: flatten a matrix into a vector
#tf.keras.layers.Dense(output size): takes the inputs provided to the model and calculates the dot product
#tf.keras.layers.Dense(output size): of the inputs and the weights and adds the bias
input_size = 10
output_size = 2     #the target is 0 or 1
hidden_layer_size = 50
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #relu is a kind of activation function
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100

max_epochs = 100

# set an early stopping mechanism
# on va utiliser l'un des paramètre de fit
# this object will monitore the validation loss and stop the training process
#the first time the validation loss starts increasing
# patience lets us decide how many consecutive increases we can tolerate
#here val_accuracy = 90%, the accuracy is good,
#the prors were 50% and 50%, so our algorithm definitely learned a lot
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)

model.fit(train_inputs,
          train_targets,
          batch_size = batch_size,
          epochs=max_epochs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose=2)


#TEST THE MODEM
#model.evaluate():return the loss value and metrics values for the model in 'test mode'
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))





