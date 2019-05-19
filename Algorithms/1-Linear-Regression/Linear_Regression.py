#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:42:10 2019

@author: jing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

class KNN():
    def __init__(self, k):
        ''' initalize the decision trees parameters '''
        self.k = k

    def predict(self, X_test, X_train, y_train):
        ypred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            Counts = np.zeros((X_train.shape[0], 2))
            for j in range(X_train.shape[0]):
                distance = np.linalg.norm(X_test[i]-X_train[j])
                Counts[j] = [distance, y_train[j]]
                # Sort distance from small to large
                knn = Counts[Counts[:,0].argsort()][:self.k]
                count = np.bincount(knn[:,1].astype('int'))
                ypred[i] = count.argmax()
        return ypred

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print('===============================Start===================================')
tic = time.process_time()

myknn = KNN(k=5)
y_pred = myknn.predict(X_test_norm, X_train_norm, y_train)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))
toc = time.process_time()

print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
