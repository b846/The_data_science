#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:03:27 2019

@author: b
S58 L413
"""

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data')

#import the relevant library
import pandas as pd

#load the data np
 #raw_data_csv_np = np.loadtxt('Absenteeism-data.csv', delimiter = ',', dtype = 'str')
raw_data_csv_df = pd.read_csv('Absenteeism-data.csv')
print(raw_data_csv_df)
pd.options.display.max_columns = None    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20
print(raw_data_csv_df)

print(raw_data_csv_df.info())
df = raw_data_csv_df.copy()

# drop ID
#the ID is a label information, we need to remove ot before analysis
#.drop delivers a temporary outputs
print(df.drop(['ID'], axis = 1))
df = df.drop(['ID'], axis = 1)

#Reason for Absence
print(df['Reason for Absence'])
print(df['Reason for Absence'].min())
print(df['Reason for Absence'].max())
print(pd.unique(df['Reason for Absence']))
print(df['Reason for Absence'].unique())
print(len(df['Reason for Absence'].unique()))
print(sorted(df['Reason for Absence'].unique()))

#get_dummies()
reason_columns = pd.get_dummies(df['Reason for Absence'])
print(reason_columns)

#check: somme des dummies variables, cela permet d'identifier le nombre de causes
reason_columns['check'] = reason_columns.sum(axis=1)
print(reason_columns)
print(reason_columns['check'].sum(axis=0))
print(reason_columns['check'].unique())

#drop of the column 0, in order to avoid multicolinearity
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
print(reason_columns)

#Group the Reasons for Absence
print(df.columns.values)
print(reason_columns.columns.values)
#on enlève la colonne 'Reasons for Absence', car ces infos sont présentes dans reason_columns
df = df.drop(['Reason for Absence'], axis = 1)
print(df)
#on va grouper les raisons des absences, afin de simplifier la régression linéaire
#on va spliter le df en plusiers dataframe
print(reason_columns.loc[:, '1':'14'])
reason_type_1 = reason_columns.loc[:, '1':'14'].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
print(reason_type_1)

# Concatenate Column Values
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
print(df)
#on renomme les colonnes
print(df.columns.values)
print(len(df.columns.values))
column_names = ['Date', 'Transportation Expense',
       'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets',
       'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
print(len(column_names))
df.columns = column_names
print(df)

#Reorder columns
columns_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
         'Date', 'Transportation Expense',
       'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets','Absenteeism Time in Hours']
print(len(columns_names_reordered))
df = df[columns_names_reordered]
print(df.head())

#Create a checkpoint
#create a copy of the current state of the df DataFrame
df_reason_modified = df.copy()
df_checkpoint = df.copy()

#Analysong the dates
print(type(df_reason_modified['Date']))
print(type(df_reason_modified['Date'][0]))
#timestamp: a classical data type for dates and time
#pd.to_datetime(): convert values into timestamp
#format %d %m %Y %H %M %S
df_reason_modified['Date'] = pd.to_datetime(df_reason_modified['Date'], format = '%d/%m/%Y')
print(df_reason_modified['Date'])
print(df_reason_modified.info())

#Extract the Month Value
print(df_reason_modified['Date'][0])
print(df_reason_modified['Date'][0].month)
list_months = []
for i in range(len(df_reason_modified)):
    list_months.append(df_reason_modified['Date'][i].month)

print(list_months)
print(len(list_months))
df_reason_modified['Month Value'] = list_months
print(df_reason_modified)


#Extract if he Day of the week
#.weekday(): returns the day of the week
print(df_reason_modified['Date'][0].weekday())

def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_modified['Day of the week'] = df_reason_modified['Date'].apply(date_to_weekday)



#Re-order the columns
print(df_reason_modified.columns.values)
print(len(df_reason_modified.columns.values))
column_names = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
       'Day of the week', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Date']
print(len(column_names))
#df_reason_date_mod = df_reason_modified[column_names]   #change l'ordre des colonnes
#df_reason_modified.columns = column_names   #cette ligne change uniquement le nom des colonnes

#remove the date column
#df_reason_modified = df_reason_date_mod.drop(['Date'], axis = 1)


#Create a checkpoint
#create a copy of the current state of the df DataFrame
df_reason_date_mod = df_reason_modified.copy()
print(df_reason_date_mod.info())

#Working on Education, Children, Pets
print(df_reason_date_mod['Education'].unique())
print(df_reason_date_mod['Education'].value_counts())
#on va combiner ceux qui ont une education 2, 3 et 4
#.map va changer les valeurs de la colonne, on va créer une dummy variable
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
print(df_reason_date_mod['Education'].unique())
print(df_reason_date_mod['Education'].value_counts())


#FINAL CHECKPOINT
df_preprocessed = df_reason_date_mod.copy()
print(df_reason_date_mod.head(10))
#Exporting the data as a *.csv file
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)



