import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numexpr as ne
import time

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
#
# plot_digit(X_train[1])
# print('Label ', y_train[1])

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

#def one_hot_encoder(y_train, y_test):
#    ''' convert label to a vector under one-hot-code fashion '''
#    from sklearn import preprocessing
#    lb = preprocessing.LabelBinarizer()
#    lb.fit(y_train)
#    y_train_ohe = lb.transform(y_train)
#    y_test_ohe = lb.transform(y_test)
#    return y_train_ohe, y_test_ohe
## label is 0 -> [1 0 0 0 0 0 0 0 0]
## label is 3 -> [0 0 0 1 0 0 0 0 0]


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


def Kernel(X, Z, kernel):
    '''kernel function '''
    if kernel == 'linear':
        K = 0.01*np.dot(X,Z.T)
    elif kernel == 'polynomial':
        r = 0
        d = 2
        K = 0.01*np.power(np.dot(X, Z.T)+ r, d)
    elif kernel == 'rbf':
        gamma = 0.001
        sigma = 1
#        K = np.zeros((X.shape[0],Z.shape[0]))
#        for i in range(X.shape[0]):
#            for j in range(Z.shape[0]):
#                tmp = (X[i,:]-Z[j,:])
#                K[i,j] = np.exp(np.dot(tmp,tmp.T)/(-gamma*sigma))
        norm_x = np.sum(X ** 2, axis = -1)
        norm_y = np.sum(Z ** 2, axis = -1)
        K = ne.evaluate('exp(- g * s * (A + B - 2 * C))', {
                'A' : norm_x[:,None],
                'B' : norm_y[None,:],
                'C' : np.dot(X, Z.T),
                'g' : gamma,
                's' : sigma
        })
    elif kernel =='sigmoid':
        gamma = 1
        r = 0
        K = np.tanh(gamma * np.dot(X, Z.T) + r)
    else:
        raise NameError('Kernel not recognized')
    return K

def predictor(X, c):
    '''predictor function '''
    return X.dot(c)

def loss(y_pred, c, y, lambd):
    M = np.where(1 - y*y_pred > 0, 1 - y*y_pred, 0)
    Loss_f =  lambd * M + np.linalg.norm(c,axis=0)
    return Loss_f

def plot_loss(loss):
    plt.xlabel('# of epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()

def Sub_Gradient_descent(X, y, c, lambd, epochs,tol):
    y = y.reshape(-1, 1)
    # convert y to a matrix nx1
    loss_history = [0]*epochs
    for epoch in range(epochs):
        learning_rate = 0.001
        yhat = predictor(X, c)
        loss_history[epoch] = loss(yhat, c, y, lambd).ravel()
        if epoch>0 and (abs(loss_history[epoch-1]-loss_history[epoch])<tol).all():
            break
        else:
            M = np.where(1 - y*yhat > 0, y, 0)
            gradient = c*(1/np.linalg.norm(c,axis=0)) - lambd * np.dot(M.T, X).T
        # updating coeffs upon the gradient change
        c = c - learning_rate*gradient
    return c, loss_history

def KernelSVM_binary_train(X, y):
   ''' Training our model based on the training data
       Input: X_train: input features
              y_train: binary labels
       Return: coeffs of the SVM model
   '''
   coeffs_0 = np.zeros((X.shape[0], 1))
   coeffs_0[0] = 1
#   coeffs_0 = np.random.rand(X_train.shape[0], 1)
   coeffs_grad, history_loss = Sub_Gradient_descent(X, y, coeffs_0, lambd=0.01, epochs=10000,tol=0.000001)
   return coeffs_grad

def KernelSVM_OVR_train(X, y, kernel):# y_train: one_hot_encoder labels
    # y_train will have 10 columns
    weights = []
    K = Kernel(X, X, kernel)
    for i in range(y.shape[1]): # 10 columns
        y_one_column = y[:,i] # pick ith columns
        weights_one_column = KernelSVM_binary_train(K, y_one_column)
        weights.append(weights_one_column)
    return weights

def prediction(weights_list, X_test, X_train, kernel):
    i = 0
    K = Kernel(X_test, X_train, kernel)
#    print(K.shape)
    for weights in weights_list:
        decision_one_column = predictor(K, weights)
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



X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

print(X_train.shape)
print(X_test.shape)

#print(y_train_ohe)

print('===============================Start===================================')
tic = time.process_time()

kernel = 'polynomial'
weights_list = KernelSVM_OVR_train(X_train_norm, y_train_ohe, kernel)
# index = 1
# plot_digit(X_test[index])
ypred = prediction(weights_list, X_test_norm,X_train_norm, kernel)
# print(ypred[index])
# print(ypred)
# print(y_test.ravel())
print('Accuracy of test data ', accuracy(ypred, y_test.ravel()))

ypred1= prediction(weights_list,X_train_norm, X_train_norm, kernel)
print('Accuracy of training data ', accuracy(ypred1, y_train.ravel()))

toc = time.process_time()
print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
