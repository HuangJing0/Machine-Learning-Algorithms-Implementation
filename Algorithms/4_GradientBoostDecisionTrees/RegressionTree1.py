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

class RegressionTrees():
    def __init__(self, max_depth=5, current_depth=1):
        ''' initalize the decision trees parameters '''
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.left_tree = None
        self.right_tree = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_y = y.shape[1]
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        if self.current_depth <= self.max_depth:
            # print('Current depth = %d' % self.current_depth)
            if len(self.y) > 0:
                self.Xy = np.concatenate((X, y), axis=1)
                self.impurity = self.impurity_calculation(self.y)
                self.best_feature_id, self.best_gain, self.best_split_value = \
                    self.find_best_split()
                if self.best_gain > 0:
                    self.split_trees()

    def predict(self, X_test):
        n_test = X_test.shape[0]
        ypred = np.zeros((n_test, self.n_y))
        for i in range(n_test):
            ypred[i] = self.tree_propogation(X_test[i])
        return ypred

    def tree_propogation(self, feature):
        if self.is_leaf_node():
#             print(self.y)
#             print('self.predic',self.predict_label())
            return self.predict_value()
#         print(feature)
        if feature[self.best_feature_id] < self.best_split_value:
            child_tree = self.left_tree
        else:
            child_tree = self.right_tree
        return child_tree.tree_propogation(feature)

    def predict_value(self):
        value = np.mean(self.y, axis = 0)
        return value


    def is_leaf_node(self):
        return self.left_tree is None

    def split_trees(self):
        # create a left tree
        self.left_tree = RegressionTrees(max_depth=self.max_depth,
                                       current_depth=self.current_depth + 1)
        self.right_tree = RegressionTrees(max_depth=self.max_depth,
                                       current_depth=self.current_depth + 1)
        best_feature_values = self.X[:, self.best_feature_id]
        left = np.array([x for x in self.Xy if x[self.best_feature_id] >= self.best_split_value])
        right = np.array([x for x in self.Xy if not x[self.best_feature_id] >= self.best_split_value])
        left_tree_X = left[:, :self.n_features]
        left_tree_y = left[:, self.n_features:]
        right_tree_X = left[:, :self.n_features]
        right_tree_y = left[:, self.n_features:]

        # fit left and right tree
        self.left_tree.fit(left_tree_X, left_tree_y)
        self.right_tree.fit(right_tree_X, right_tree_y)


    def find_best_split(self):
        best_feature_id = None
        best_gain = 0
        best_split_value = None
        for feature_id in range(self.n_features):
#         for feature_id in range(1, 2):
            current_gain, current_split_value = self.find_best_split_one_feature(feature_id)
            if current_gain is None:
                continue
            if best_gain < current_gain:
                best_feature_id = feature_id
                best_gain = current_gain
                best_split_value = current_split_value
        return best_feature_id, best_gain, best_split_value

    def find_best_split_one_feature(self, feature_id):
        '''
            Return information_gain, split_value
        '''
        feature_values = self.X[:, feature_id]
        unique_feature_values = np.unique(feature_values)
        best_gain = 0.0
        best_split_value = None
        if len(unique_feature_values) == 1:
            return best_gain, best_split_value
        for fea_val in unique_feature_values:
            left = np.array([x for x in self.Xy if x[feature_id] >= fea_val])
            right = np.array([x for x in self.Xy if not x[feature_id] >= fea_val])
            if len(left) > 0 and len(right) > 0:
                left_tree_X = left[:, :self.n_features]
                left_tree_y = left[:, self.n_features:]
                right_tree_X = left[:, :self.n_features]
                right_tree_y = left[:, self.n_features:]
                left_impurity = self.impurity_calculation(left_tree_y)
                right_impurity = self.impurity_calculation(right_tree_y)
    #             print(left_GINI)
                # calculate gain
                left_n_samples = left_tree_X.shape[0]
                right_n_samples = right_tree_X.shape[0]
                current_gain = sum(self.impurity - (left_n_samples/self.n_samples * left_impurity + \
                                                        right_n_samples/self.n_samples * right_impurity))
    #             print(self.GINI)
    #             print(self.GINI)
                if best_gain < current_gain:
                    best_gain = current_gain
                    best_split_value = fea_val
            return best_gain, best_split_value

    def impurity_calculation(self, y):
        mean = np.ones(np.shape(y)) * y.mean(0)
        n_samples = np.shape(y)[0]
        impurity = (1 / n_samples) * np.diag((y - mean).T.dot(y - mean))
        return impurity

def PCC(y_pred, y_test):
    from scipy import stats
    a = y_test
    b = y_pred
    pcc = stats.pearsonr(a, b)
    return pcc

def RMSE(ypred, yexact):
    diff = ypred - yexact
    return np.sqrt(sum(diff*diff)/ypred.shape[0])

X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('===============================Start===================================')
tic = time.process_time()
mydt = RegressionTrees(max_depth=11)
mydt.fit(X_train_norm, y_train)
y_pred = mydt.predict(X_test_norm)
print('RMSE of our model ', RMSE(y_pred, y_test),'\n')
print('PCC of our model ', PCC(y_pred, y_test))
toc = time.process_time()

print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
