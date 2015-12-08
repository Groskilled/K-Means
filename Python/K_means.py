#!/usr/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def plotClusters(X, idx):
    for i in range(X.shape[0]):
        if (idx[i] == 1):
            plt.plot(X[i, 0], X[i, 1], 'ro');
        elif (idx[i] == 2):
            plt.plot(X[i, 0], X[i, 1], 'bo');
        elif (idx[i] == 3):
            plt.plot(X[i, 0], X[i, 1], 'go');
        elif (idx[i] == 4):
            plt.plot(X[i, 0], X[i, 1], 'yo');
    plt.show()

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
K = 4
centroids = init_centroids(X, K)
ret = findClosestCentroids(X, centroids)
tmpCentroids = computeCentroids(X, ret, K)
plotClusters(X, ret)
while (1):
    ret = findClosestCentroids(X, centroids)
    tmpCentroids = computeCentroids(X, ret, K)
    if (tmpCentroids == centroids).all():
        break
    else:
        centroids = tmpCentroids
        plotClusters(X, ret)
print "This is the solution found.\n"
plotClusters(X, ret)
