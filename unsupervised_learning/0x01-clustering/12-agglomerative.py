#!/usr/bin/env python3
""" doc """

import scipy.cluster.hierarchy
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ doc """
    Z = scipy.cluster.hierarchy.linkage(X,
                                        method='ward')
    dendrogram = scipy.cluster.hierarchy.dendrogram(Z,  color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z,
                                            t=dist,
                                            criterion='distance')
    plt.show()
    return(clss)
