#!/usr/bin/env python3
""" doc """
import numpy as np


def initialize(X, k):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None

    minimum = np.amin(X, axis=0)
    maximum = np.amax(X, axis=0)

    return np.random.uniform(minimum, maximum, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    centroids = initialize(X, k)
    labels = None
    for i in range(iterations):
        centroids_cpy = np.copy(centroids)
        distance = np.linalg.norm(X[:, None] - centroids, axis=-1)
        labels = np.argmin(distance, axis=-1)
        for j in range(k):
            index = np.argwhere(labels == j)
            if not len(index):
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = np.mean(X[index], axis=0)
        if (centroids_cpy == centroids).all():
            return centroids, labels

    distance = np.linalg.norm(X[:, None] - centroids, axis=-1)
    labels = np.argmin(distance, axis=-1)

    return centroids, labels
