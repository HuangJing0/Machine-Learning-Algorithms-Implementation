#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:11:51 2019

@author: jing
"""

import math
import time
import random
import numpy as np
import pandas as pd
tic = time.clock()

np.random.seed(1)


def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values  # convert values in dataframe to numpy array (features)
    y = df_y.values  # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler  # import libaray
    scaler = StandardScaler()  # call an object function
    scaler.fit(X_train)  # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train)  # apply normalization on X_train
    # we use the same normalization on X_test
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm


# def one_hot_encoder(y_train, y_test):
#     ''' convert label to a vector under one-hot-code fashion '''
#     from sklearn import preprocessing
#     lb = preprocessing.LabelBinarizer()
#     lb.fit(y_train)
#     y_train_ohe = lb.transform(y_train)
#     y_test_ohe = lb.transform(y_test)
#     return y_train_ohe, y_test_ohe

def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe1 = lb.transform(y_train)
    y_test_ohe1 = lb.transform(y_test)
    y_train_ohe = np.where(y_train_ohe1 > 0, y_train_ohe1, -1)
    y_test_ohe = np.where(y_test_ohe1 > 0, y_test_ohe1, -1)
    return y_train_ohe, y_test_ohe


X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train_ohe.shape)
print(y_test_ohe.shape)


class kernelSVM():
    def __init__(self, X, y, lr=0.01, lambd=0.01):
        '''
        Input:
        - X: shape (N, D)
        - y: shape (N,C) C = 10
        - W shape (D,C)
        '''
        self.X = X
        self.y = y
        self.m = X.shape[0]  # numbers of samples
        self.n = X.shape[1]  # numbers of features
        self.p = y.shape[1]
        self.W = np.random.rand(self.m, self.p)  # 784 by 10
        self.b = np.zeros((1, self.p))   # 1 by 10
        self.lr = lr
        self.lambd = lambd
        #self.K = K

    def forward(self):
        self.y_hat = np.dot(self.X, self.W) + self.b # 500 by 10
        self.cond = 1 - self.y * self.y_hat

    def loss(self):
        self.forward()
        self.hinge_loss = np.where(self.cond > 0, self.y, 0)  # 2000 by 10
        self.loss = np.linalg.norm(self.W, axis=0) + self.lambd * self.hinge_loss


    def sub_gradient_descent(self):
        y = np.where(self.cond > 0, self.y, 0)
        y_b = np.where(self.cond > 0, 1 - self.y, 0)
        dW = np.linalg.norm(self.W, axis=0) * self.W - \
            self.lambd * np.dot(y.T, self.X).T
        db = np.sum(self.lambd * y_b ,axis = 0)
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db
        print(y)
        time.sleep(.5)


    def predict(self, K_test):
        y_hat_test = np.dot(K_test, self.W) + self.b
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_test_samples = K_test.shape[0]
        ypred = np.zeros(num_test_samples, dtype=int)
        for i in range(num_test_samples):
            ypred[i] = labels[np.argmax(y_hat_test[i, :])]
        return ypred


def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype=int)
    return np.sum(p) / float(len(yexact))

def Kernel(X, Z, kernel):
    '''kernel function '''
    if kernel == 'linear':
        K = np.dot(X,Z.T)
    elif kernel == 'polynomial':
        r = 0
        d = 1
        K = np.power(np.dot(X, Z.T)+ r, d)
    elif kernel == 'rbf':
        gamma = 2
        sigma = 3
        K = np.zeros((X.shape[0],Z.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Z.shape[0]):
                tmp = (X[i,:]-Z[j,:])
                K[i,j] = np.exp(np.dot(tmp,tmp.T)/(-gamma*sigma))
    elif kernel =='sigmoid':
        gamma = 1
        r = 0
        K = np.tanh(gamma * np.dot(X, Z.T) + r)
    else:
        raise NameError('Kernel not recognized')
    return K

kernel = 'linear'
K_train = Kernel(X_train_norm, X_train_norm, kernel)
mySVM = kernelSVM(K_train, y_train_ohe, lr=0.01, lambd=0.01)

epoch_num = 200
for i in range(epoch_num):
    mySVM.forward()
    mySVM.sub_gradient_descent()


K_test =Kernel(X_test_norm, X_train_norm, kernel)
y_pred = mySVM.predict(K_test)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.clock()

print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
