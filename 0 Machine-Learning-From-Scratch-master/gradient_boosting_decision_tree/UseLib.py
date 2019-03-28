#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:17:44 2019

@author: jing
"""

import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier



def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

def main():

    print ("-- Gradient Boosting Classification --")

    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", y_train.shape)

    clf = GradientBoostingClassifier(n_estimators=1,max_depth=3)
    clf
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    Accuracy = accuracy(y_pred, y_test)

    print ("Accuracy:", Accuracy)


if __name__ == "__main__":
    main()
