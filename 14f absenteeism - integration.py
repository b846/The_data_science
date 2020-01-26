#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 01:11:47 2019

@author: b
Absenteism Exercise - Integration
les fichiers 'model', 'scaler' et csv doivent être présents dans le repertoire de travail
this file is currently not working
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#importation des bibliotèques
from absenteeism_module import *

#load the data np
print(pd.read_csv('Absenteeism_new_data.csv'))

# création du model
#absenteeism_model: the name of the class that has been created
#model: we must create an instance of this class
#scaler: it contains the statistical parameters needed to adjust the magnitude of all numbers
model = absenteeism_model('model', 'scaler') 

#load the data
model.load_and_clean_data('Absenteeism_preprocessed.csv')       #will load and preprocess the entire data set we provide
#.predict_outputs: its role is to fed the cleaned data into the model, and deliver the outputs we discussed (ie the probability of being absent)
model.predict_outputs()

#Export our model
model.predicted_outputs().to_csv(‘Absenteeism_predictions.csv', index = False)


