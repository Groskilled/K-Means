#!/usr/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def init_centroids(X, K):
    idx = random.sample(range(X.shape[0]), K)
    ret = []
    for i in range(K):
        ret.append(X[idx[i]])
    return ret

def extractRow(idx, X):
    ret = []
    for i in range(idx.shape[0]):
        if idx[i]:
            ret.append(X[i])
    return np.array(ret)

def findClosestCentroids(X, centroids):
    idx = np.ones((X.shape[0], 1))
    for i in range(X.shape[0]):
        diffs = X[i] - centroids
        idx[i] = np.argmin(np.sum(np.square(diffs), axis=1)) + 1
    return idx

def computeCentroids(X, idx, K):
    m,n = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i] = np.mean(extractRow(idx == (i + 1), X), axis=0)
    return centroids

X = sio.loadmat('ex7data2.mat')
X = np.array(X['X'])
plt.plot(X[:, 0], X[:, 1], 'ro')
plt.show()
K = 3
centroids = np.array([[3, 3], [6, 2], [8, 5]])
ret = findClosestCentroids(X, centroids)
print computeCentroids(X, ret, K)
#centroids = init_centroids(X, K)
