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

from ClassificationTree import ClassificationTrees

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


class RandomForest():
    def __init__(self, n_estimators = 100, min_samples_split = 2, min_gain = 0,
                 max_depth = float("inf"), max_features = None):
        ''' initalize the decision trees parameters float("inf") '''
        self.max_depth = max_depth
        self.n_estimators = n_estimators # Number of trees
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_gain = min_gain

        #Build sequentially N trees
        self.trees = []
        for i in range(self.n_estimators):
            #self.trees.append(RegressionTree(min_impurity=self.min_impurity, max_depth=self.max_depth))
            self.trees.append(ClassificationTrees(max_depth = self.max_depth))

    def fit(self, X, y):
        samples = self.bootstrap_sample(X, y)
        n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            X_sample, y_sample = samples[i]
            id = np.random.choice(n_features, self.max_features, replace=True)
            X_sample = X_sample[:, id]
            self.trees[i].fit(X_sample, y_sample)
            self.trees[i].feature_id = id
            if i%10 == 0:
                print("Tree", i, "fit complete")

    def predict(self, X):
        ypreds = []
        for i in range(self.n_estimators):
            id = self.trees[i].feature_id
            X_sample = X[:, id]
            ypre = self.trees[i].predict(X_sample)
            ypreds.append(ypre)
        ypreds = np.array(ypreds).T
        ypred = []
        for y in ypreds:
            ypred.append(np.bincount(y.astype('int')).argmax())
        return ypred

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        y = y.reshape(n_samples, 1)

        X_y = np.hstack((X, y))
        np.random.shuffle(X_y)

        samples = []
        for i in range(self.n_estimators):
            id = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_X_y = X_y[id, :]
            bootstrap_X = bootstrap_X_y[:, :-1]
            bootstrap_y = bootstrap_X_y[:, -1:]
            samples.append([bootstrap_X, bootstrap_y])
        return samples

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

myRandomForest = RandomForest(n_estimators=200, max_depth = 6)
myRandomForest.fit(X_train_norm, y_train)
y_pred = myRandomForest.predict(X_test_norm)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.process_time()
print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
