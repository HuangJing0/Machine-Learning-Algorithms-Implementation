#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:20:38 2019

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

def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance

def impurity_calculation(y, y1, y2):
    LeftEntropy = len(y1) * calculate_variance(y1)
    RightEntropy = len(y2) * calculate_variance(y2)
    impurity = calculate_variance(y) - (LeftEntropy + RightEntropy)/len(y)
    return sum(impurity)

class SaveTree():
    def __init__(self, feature_id = None, split_value = None,\
                 leaf_value = None, left_tree = None, right_tree = None):
        self.feature_id = feature_id
        self.split_value = split_value
        self.leaf_value = leaf_value
        self.left_tree = left_tree
        self.right_tree = right_tree

class RegressionTrees(object):
    def __init__(self, max_depth, min_impurity = 1e-7, min_samples_split = 2):
        ''' initalize the decision trees parameters '''
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self.create_tree(X, y)

    def create_tree(self, X, y, current_depth = 0):
        '''
        Creat tree by recursive
        '''
        n_features = X.shape[1]
        n_samples = X.shape[0]
        best_feature_id = None
        best_gain = 0
        best_split_value = None
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_id in range(n_features):
                feature_values = np.expand_dims(X[:, feature_id], axis = 1)
                unique_feature_values = np.unique(feature_values)

                for fea_val in unique_feature_values:
                    left = np.array([x for x in Xy if x[feature_id] >= fea_val])
                    right = np.array([x for x in Xy if not x[feature_id] >= fea_val])

                    if len(left) > 0 and len(right) > 0:
                        left_y = left[:, n_features:]
                        right_y = right[:, n_features:]
                        impurity = impurity_calculation(y, left_y, right_y)

                        if best_gain < impurity:
                            best_feature_id = feature_id
                            best_gain = impurity
                            best_split_value = fea_val
                            left_tree_X = left[:, :n_features]
                            left_tree_y = left[:, n_features:]
                            right_tree_X = right[:, :n_features]
                            right_tree_y = right[:, n_features:]



        if best_gain > self.min_impurity:
            left_tree = self.create_tree(left_tree_X, left_tree_y, current_depth + 1)
            right_tree = self.create_tree(right_tree_X, right_tree_y, current_depth + 1)
            return SaveTree(feature_id = best_feature_id, split_value = best_split_value,\
                        left_tree = left_tree, right_tree = right_tree)


        value = np.mean(y, axis=0)
        return SaveTree(leaf_value = value)


    def predict_value(self, x, tree = None):
        if tree is None:
            tree = self.root
        if tree.leaf_value is not None:
            return tree.leaf_value
        feature_values = x[tree.feature_id]
        if feature_values >= tree.split_value:
             return self.predict_value(x, tree.left_tree)
        else:
             return self.predict_value(x, tree.right_tree)

    def predict(self, X):
        ypred = []
        for x in X:
            ypred.append(self.predict_value(x))
        return np.asarray(ypred)

# def PCC(y_pred, y_test):
#     from scipy import stats
#     a = y_test
#     b = y_pred
#     pcc = stats.pearsonr(a, b)
#     return pcc
#
# def RMSE(ypred, yexact):
#     diff = ypred - yexact
#     # print(diff.shape)
#     return np.sqrt(sum(diff*diff)/len(ypred))
#
# X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
# X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
# X_train_norm, X_test_norm = normalize_features(X_train, X_test)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print('===============================Start===================================')
# tic = time.process_time()
# mydt = RegressionTrees(max_depth=11)
# mydt.fit(X_train_norm, y_train)
# y_pred = mydt.predict(X_test_norm)
# print('RMSE of our model ', RMSE(y_pred, y_test),'\n')
# print('PCC of our model ', PCC(y_pred, y_test))
# toc = time.process_time()
#
# print('Totol time:' + str((toc - tic)) + 's')
# print('===============================Finish===================================')
