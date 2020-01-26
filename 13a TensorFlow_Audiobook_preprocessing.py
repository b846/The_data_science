#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:39:01 2019

@author: b
more advanced model
Inputs
Linear combinaison xw + b
sigmoid function (non-linearity)
outputs
the layer is the building block if neural networks
when we have more than one layer, we are talking about a deep neural network
train_images[train_images >= .5] = 1     binarization
""" 

   
# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#import the relevant library
import numpy as np
from sklearn import preprocessing

#load the data
raw_csv_data = np.loadtxt('Audiobooks-data.csv', delimiter = ',')
unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]    #last column

#BALANCE THE DATA SETS
num_one_targets = int(np.sum(targets_all))      #nombre  de client ayant de nv acheté après 6 mois
zero_targets_counter = 0
indices_to_remove = []

#on parcout toutes les targets
#si la target est nulle, et le nb de 0 est plus gd que le nb de 1, alors on enlève la valeur
for i in range(targets_all.shape[0]):    #targets_all.shape[0]:length of the vector
    if targets_all[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
            
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

#STANDARDIZE THE INPUTS
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

#shuffle the data
#on mélange les données par le shuffle
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


#split the data into train, validation and test
samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

#the last of these 3 numbers shoulb be around 50%
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

#Save the three datasets in *.npz
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)