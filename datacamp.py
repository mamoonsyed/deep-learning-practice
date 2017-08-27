# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

white = pd.read_csv("http://archive.ics.uci.edu/ml/\
                    machine-learning-databases/wine-quality/\
                    winequality-white.csv",
                    sep=';') # Read in white wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/\
                  machine-learning-databases/wine-quality/\
                  winequality-red.csv",
                  sep=';') # Read in red wine data 


red['type'] = 1 # Add `type` column to `red` with value 1
white['type'] = 0 # Add `type` column to `white` with value 0
wines = red.append(white, ignore_index=True) # Append `white` to `red`


X=wines.ix[:,0:11] # Specify the data 
y=np.ravel(wines.type) # Specify the target labels and flatten the array 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42) # Split the data up in train and test sets


scaler = StandardScaler().fit(X_train) # Define the scaler 
X_train = scaler.transform(X_train) # Scale the train set
X_test = scaler.transform(X_test) # Scale the test set