#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:11:51 2019

@author: jing
"""

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y

def plot_digit(feature_vector):
    plt.gray()
    plt.matshow(feature_vector.reshape(28,28))
    plt.show()

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm


def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe
# label is 0 -> [1 0 0 0 0 0 0 0 0]
# label is 3 -> [0 0 0 1 0 0 0 0 0]

#print(y_train_ohe)

def predictor(X, c):
    '''predictor function '''
    return X.dot(c)

def loss(y_pred, c, y, lamda):
    S = np.where(1 - y*y_pred > 0, 1 - y*y_pred, 0)
    Loss_f =  lamda * S + np.linalg.norm(c,axis=0)
    return Loss_f

def plot_loss(loss):
    plt.xlabel('# of epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()

def Sub_Gradient_descent(X, y, c, lamda, epochs, learning_rate):
    y = y.reshape(-1, 1)
    # convert y to a matrix nx1
    loss_history = [0]*epochs
    for epoch in range(epochs):
        yhat = predictor(X, c)
        loss_history[epoch] = loss(yhat, c, y, lamda).ravel()
        M = np.where(1 - y*yhat > 0, y, 0)
        gradient = c*(1/np.linalg.norm(c)) - lamda * np.dot(M.T, X).T
        # updating coeffs upon the gradient change
        c = c - learning_rate*gradient
    return c, loss_history

def SVM_binary_train(X_train, y_train):
    ''' Training our model based on the training data
        Input: X_train: input features
               y_train: binary labels
        Return: coeffs of the SVM model
    '''
    coeffs_0 = np.zeros((X_train.shape[1], 1))
    coeffs_0[0] = 1
#    coeffs_0 = np.random.rand(X_train.shape[1], 1)
    coeffs_grad, history_loss = Sub_Gradient_descent(X_train, y_train, coeffs_0, lamda=0.01, epochs=200, learning_rate=0.0001)
    return coeffs_grad

def SVM_OVR_train(X_train, y_train):# y_train: one_hot_encoder labels
    # y_train will have 10 columns
    weights = []
    for i in range(y_train.shape[1]): # 10 columns
        y_train_one_column = y_train[:,i] # pick ith columns
        weights_one_column = SVM_binary_train(X_train, y_train_one_column)
        weights.append(weights_one_column)
    return weights


def prediction(weights_list, X_test):
    i = 0
    for weights in weights_list:
        decision_one_column = predictor(X_test, weights)
        # probabily of one column
        if i == 0:
            decision_matrix = decision_one_column
        else:
            # combine all decision columns to form a matrix
            decision_matrix = np.concatenate(
                              (decision_matrix, decision_one_column),
                               axis=1)
        i += 1
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_test_samples = X_test.shape[0]
    # find which index gives us the highest probability
    ypred = np.zeros(num_test_samples, dtype=int)
    for i in range(num_test_samples):
        ypred[i] = labels[np.argmax(decision_matrix[i,:])]
    return ypred

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

# =========================================================================

X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)
print(X_train.shape)
print(X_test.shape)

#==========================================================================
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(X_train_norm, y_train.ravel())
y_hat = clf.predict(X_test_norm)
print('Accuracy of library model ', accuracy(y_hat, y_test.ravel()))
#==========================================================================
print('===============================Start===================================')
tic = time.process_time()
weights_list = SVM_OVR_train(X_train_norm, y_train_ohe)
ypred = prediction(weights_list, X_test_norm)
toc = time.process_time()
print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
# index = 0
# plot_digit(X_test[index])
# print(ypred[index])
print('Accuracy of our model ', accuracy(ypred, y_test.ravel()))
