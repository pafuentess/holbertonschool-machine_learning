#!/usr/bin/env python3
""" doc """

import sklearn.cluster


def kmeans(X, k):
    """ doc """
    kmean = sklearn.cluster.KMeans(k)
    kmean.fit(X)
    return kmean.cluster_centers_, kmean.labels_
