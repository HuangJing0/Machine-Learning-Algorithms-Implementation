from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
# Import helper functions
from utils import train_test_split, accuracy_score
from utils.loss_functions import CrossEntropy
from utils import Plot
from gradient_boosting_decision_tree.gbdt_model import GBDTClassifier


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


def Accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

def main():

    print ("-- Gradient Boosting Classification --")

    X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
    X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    mybgdt =GBDTClassifier()
    mybgdt.fit(X_train_norm, y_train.ravel())
    y_pred = mybgdt.predict(X_test_norm)
    print('Accuracy of our model ', Accuracy(y_pred, y_test.ravel()))

#    data = datasets.load_iris()
#    X = data.data
#    y = data.target
#    
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#    print(y_train)
#
#    clf = GBDTClassifier()
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#
#    accuracy = accuracy_score(y_test, y_pred)
#
#    print ("Accuracy:", accuracy)

    accuracy = Accuracy(y_pred, y_test.ravel())
    Plot().plot_in_2d(X_test, y_pred,
        title="Gradient Boosting",
        accuracy=accuracy)


if __name__ == "__main__":
    main()
