
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

class Kmeans():
    def __init__(self, k = 2, n_interation = 500):
        self.k = k
        self.n_interation = n_interation

    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        """
        tmp = np.sum((p1-p2)**2)
        return np.sqrt(tmp)

    def rand_Cluster_Centroid(self, X):
        """Generate k center within the range of data set."""
        n_features = X.shape[1]
        centroids = np.zeros((self.k,n_features))
        for i in range(n_features):
            dmin, dmax = np.min(X[:,i]), np.max(X[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(self.k)
        return centroids

    def if_converge(self,centroids1, centroids2):
        # if centroids not changed, we say 'converged'
         set1 = set([tuple(c) for c in centroids1])
         set2 = set([tuple(c) for c in centroids2])
         return (set1 == set2)

    def Find_closest_centroid(self, sample, centroids):
        """ Return the index of the closest centroid to the sample """
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = np.linalg.norm(sample-centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def predict(self, X):
        """
        repeate N times
        """
        centroids = self.rand_Cluster_Centroid(X)
        n_samples, n_features = X.shape
        ypred = np.zeros((n_samples,1))
        converge = False

        for i in range(self.n_interation):
            clusters = [[] for i in range(self.k)]
            for sample_i, sample in enumerate(X):
                centroid_i = self.Find_closest_centroid(sample, centroids)
                clusters[centroid_i].append(sample_i)
            old_centroids = centroids
            centroids = np.zeros((self.k, n_features))
            for i, cluster in enumerate(clusters):
                centroid = np.mean(X[cluster], axis=0)
                centroids[i] = centroid
            converge = self.if_converge(old_centroids,centroids)
            if converge:
                break

        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                ypred[sample_i] = cluster_i
        return ypred, centroids

def Cluster_show(X, centroids, ypred):
    X0 = X[np.where(ypred==0)[0],:]
    X1 = X[np.where(ypred==1)[0],:]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.scatter(X[:,0],X[:,1],c='c',s=30,marker='o')
    ax2.scatter(X0[:,0],X0[:,1],c='r')
    ax2.scatter(X1[:,0],X1[:,1],c='y')
    ax2.scatter(centroids[:,0],centroids[:,1],c='b',s=120,marker='o')
    plt.show()


X_train, y_train = read_dataset('Iris_X_train.csv', 'Iris_y_train.csv')
X_test, y_test = read_dataset('Iris_X_test.csv', 'Iris_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)


X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)
print(X.shape)

print('===============================Start===================================')
tic = time.process_time()

myKmeans = Kmeans(k = 2, n_interation = 200)
ypred, centroids= myKmeans.predict(X)

toc = time.process_time()
print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')

Cluster_show(X, centroids, ypred)
# Cluster_show(X, centroids, y)
