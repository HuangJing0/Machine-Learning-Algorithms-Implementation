#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:42:10 2019

@author: jing
"""

import pandas as pd
import numpy as np

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y

X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')

print('Shape of X_train ', X_train.shape)
print('Shape of X_test ', X_test.shape)

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

X_train_norm, X_test_norm = normalize_features(X_train, X_test)
print(X_test_norm)

class LinearRegression:
    def __init__(self, n_iter=3000, lr=0.01, Lambda = 0.1):
        self.n_iter = n_iter
        self.lr = lr
        self.Lambda = Lambda

    def initialized_weight(self, m):
        limit = np.sqrt(1/m)
        w = np.random.uniform(-limit, limit, (m, 1))
        b = 0
        self.w = np.insert(w,0,b,axis=0)

    def fit(self, X, y):
        n, m = X.Shape # m: number of features, n: number of samples
        self.initialized_weight(m)
        X = np.insert(X,0,1,axis=0)
        y = np.reshape(y, (n,1))
        self.train_error = []

    def gradident_descent(self,):
        for i in range(self.n_iter):
            y_pred = X.dot(self.w)
            current_loss = self.loss()
            self.train_error.append(current_loss)
            print(current_loss)
            w_grad = X.T.dot(y_pred - y) + self.Lambda*self.w
            self.w = self.w - self.lr*w_grad
            
    def predict():
