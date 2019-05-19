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
from scipy.special import logsumexp

from RegressionTree import RegressionTrees

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

def one_hot_encoder(y):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y_ohe1 = lb.transform(y)
#     y_train_ohe1 = lb.transform(y_train)
#     y_test_ohe1 = lb.transform(y_test)
# #    y_train_ohe = np.where(y_train_ohe1 > 0, y_train_ohe1, -1)
# #    y_test_ohe = np.where(y_test_ohe1 > 0, y_test_ohe1, -1)
    return y_ohe1

def NegativeGradient(y, ypred, k):
    return  y[:,k] - np.nan_to_num(np.exp(ypred[:,k] - logsumexp(ypred, axis=1)))

class GBDT():
    def __init__(self, max_depth, min_impurity,eta, n_estimators):
        ''' initalize the decision trees parameters '''
        self.max_depth = max_depth
        self.n_estimators = n_estimators # Number of trees
        self.min_impurity = min_impurity
        self.eta = eta

        #Build sequentially N trees
        self.trees = []
        for i in range(self.n_estimators):
            #self.trees.append(RegressionTree(min_impurity=self.min_impurity, max_depth=self.max_depth))
            self.trees.append(RegressionTrees(max_depth = self.max_depth))

    def fit(self, X, y):
        #Under one-hot-code fashion, y.shape[1] states the # of classes
        self.n_classes = y.shape[1]
        self.trees[0].fit(X, y)
        ypred = self.trees[0].predict(X)
        for i in range(1, self.n_estimators):
            residual = np.array([NegativeGradient(y, ypred, k) for k in range(self.n_classes)]).T
            #pred = y - residual
            # print(residual.shape)
            self.trees[i].fit(X, residual)
            ypred += np.multiply(self.eta, self.trees[i].predict(X))
            # # One-step approximate Newton update : yk_(n+1) = yk_n + sum(yk_n - pk)/sum(pk(1-pk))
            # yn = self.trees[i].predict(X)
            # ynT = yn.tolist()
            # terminal_region = np.unique(yn, axis=0)
            # self.learning_rate = np.zeros(())
            # for k in range(self.n_classes):
            #     for j in range(terminal_region.shape[0]):
            #         index = np.unique(np.where(ynT == terminal_region[j,:])[0])
            #         resi = residual.take(index, axis = 0)
            #         print(index.shape)
            #         # time.sleep(5)
            #         numerator = np.sum(resi)
            #         numerator *= (self.n_classes - 1) / self.n_classes
            #         denominator = np.sum(np.abs(resi) * (1.0 - np.abs(resi)))
            #         if denominator == 0.0:
            #             learning_rate = 0.0
            #         else:
            #             learning_rate = numerator / denominator
            # ypred += self.eta * learning_rate
            print("Tree", i, "fit complete")


    def predict(self, X_test):
        ypred = self.trees[0].predict(X_test)
        for i in range(1, self.n_estimators):
            # # One-step approximate Newton update
            # yn = self.trees[i].predict(X_test)
            # for k in range(self.n_classes):
            #     resi = NegativeGradient(y, yn, k)
            #     numerator = np.sum(resi)
            #     numerator *= (self.K - 1) / self.K
            #     denominator = np.sum((y[:,k] - resi) * (1.0 - y[:,k] + resi))
            #     if denominator == 0.0:
            #         ypred[k] += 0.0
            #     else:
            #         ypred[k] += numerator / denominator
            ypred += np.multiply(self.eta, self.trees[i].predict(X_test))
        # Using Softmax function to get the probability
        ypred = np.exp(ypred) / np.expand_dims(np.sum(np.exp(ypred), axis=1), axis=1)
        # Set label to the value that maximizes probability
        ypred = np.argmax(ypred, axis=1)
        return ypred

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe = one_hot_encoder(y_train)
y_test_ohe = one_hot_encoder(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('===============================Start===================================')
tic = time.process_time()
mybgdt = GBDT(max_depth = 3,eta = 1, min_impurity=1e-7, n_estimators=20)
mybgdt.fit(X_train_norm, y_train_ohe)
y_pred = mybgdt.predict(X_test_norm)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))
toc = time.process_time()

print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
