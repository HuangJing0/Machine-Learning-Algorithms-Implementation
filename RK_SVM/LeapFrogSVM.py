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

# np.seterr(divide='ignore', invalid='ignore')

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



def plot_loss(loss):
    plt.xlabel('# of epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()

def predictor(X, c):
    '''predictor function '''
    return np.dot(X, c)

class Runge_Kutta_SVM():
    def __init__(self, X_train, y_train, lamda, epochs, learning_rate, tolerance = 1e-6, n = 100):
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.tolerance = tolerance
        self.epochs = epochs
        self.n = n

    def loss(self, y_pred, c, y):
        S = np.where(1 - y*y_pred > 0, 1 - y*y_pred, 0)
        Loss_f =  self.lamda * S + np.linalg.norm(c,axis=0)
        return Loss_f

    def Sub_Gradient(self, y, c):
        y = y.reshape(-1, 1)
        yhat = predictor(self.X_train, c)
        M = np.where(1 - y*yhat > 0, y, 0)
        gradient = c*(1/np.linalg.norm(c)) - self.lamda * np.dot(M.T, self.X_train).T
        return gradient

    # def F(self, y, c1, c0):
    #     gradient = self.Sub_Gradient(y, c1)
    #     return c1 - c0 - self.learning_rate * gradient

    # def Steffensen(self, y, c):
    #     iter = 0
    #     c0 = c + self.F(y,c,c)
    #     F = self.F(y, c0, c)
    #     while any(abs(F) > self.tolerance) and iter < self.n:
    #         F = self.F(y, c0, c)
    #         G = self.F(y, c0+F, c)/F - 1
    #         c_temp = c0 - F/G
    #         c0 = c_temp
    #         iter += 1
    #     print(iter)
    #     return c0

    def Runge_Kutta(self, y, c1, c0):
        loss_history = [0]*self.epochs
        yhat = predictor(self.X_train, c0)
        loss_history[1] = self.loss(y, c0, yhat).ravel()
        # print('epoch = ',1)
        # print(loss_history[1])
        yhat = predictor(self.X_train, c1)
        loss_history[2] = self.loss(y, c1, yhat).ravel()
        # print('epoch = ',2)
        # print(loss_history[2])
        for epoch in range(3,self.epochs):
            gradient = self.Sub_Gradient(y, c1)
            c = c0 - 2*self.learning_rate*(gradient)
            yhat = predictor(self.X_train, c)
            loss_history[epoch] = self.loss(y, c, yhat).ravel()
            # print('epoch =', epoch)
            # print(loss_history[epoch])
            c0 = c1
            c1 = c
        return c,loss_history

    def SVM_binary_train(self, y_train_one_column):
        ''' Training our model based on the training data
            Input: X_train: input features
                   y_train: binary labels
            Return: coeffs of the SVM model
        '''
        coeffs_0 = np.zeros((self.X_train.shape[1], 1))
        coeffs_0[0] = 1
        # coeffs_0 = np.random.rand(X_train.shape[1], 1)
        gradient = self.Sub_Gradient(y_train_one_column, coeffs_0)
        coeffs_1 = coeffs_0 - self.learning_rate * gradient
        coeffs_grad, history_loss = self.Runge_Kutta(y_train_one_column, coeffs_0,coeffs_1)
        return coeffs_grad

    def SVM_OVR_train(self):# y_train: one_hot_encoder labels
        # y_train will have 10 columns
        self.weights_list = []
        for i in range(self.y_train.shape[1]): # 10 columns
            y_train_one_column = self.y_train[:,i] # pick ith columns
            weights_one_column = self.SVM_binary_train(y_train_one_column)
            self.weights_list.append(weights_one_column)

    def prediction(self, X_test):
        i = 0
        for weights in self.weights_list:
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

# def gradient_descent(X, y, c, epochs=1000, learning_rate=0.0001):
#     y = y.reshape(-1, 1) # convert y to a matrix nx1
#     loss_history = [0]*epochs
#     for epoch in range(epochs):
#         yhat = predictor(X, c)
#         loss_history[epoch] = loss(y, yhat).ravel()
#         XT = X.T
#         gradient = XT.dot(yhat - y)/float(len(y))
#         # updating coeffs upon the gradient change
#         c = c - learning_rate*gradient
#     return c, loss_history

# # Stochastic GD (SGD)
# def SGD(X, y, c, epochs=1000, learning_rate=0.00, batch_size=10):
#     y = y.reshape(-1, 1)
#     loss_history = [0]*epochs
#     for epoch in range(epochs):
#         # loop through batches
#         batch_loss = []
#         for i in np.arange(0, X.shape[0], batch_size):
#             X_current_batch = X[i:i+batch_size]
#             y_current_batch = y[i:i+batch_size]
#             yhat = predictor(X_current_batch, c)
#             loss_current_batch = loss_function(X_current_batch, c, y_current_batch).ravel()
#             batch_loss.append(loss_current_batch)
#             gradient = XT.dot(yhat - y)/float(len(y))
#             c = c - learning_rate*gradient
#         loss_history[epoch] = np.average(batch_loss)
#     return c, loss_history

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
mySVM = Runge_Kutta_SVM(X_train_norm, y_train_ohe, lamda=0.01, epochs=100, learning_rate=0.01)
mySVM.SVM_OVR_train()
ypred = mySVM.prediction(X_test_norm)
toc = time.process_time()
print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
# index = 0
# plot_digit(X_test[index])
# print(ypred[index])
print('Accuracy of our model ', accuracy(ypred, y_test.ravel()))
