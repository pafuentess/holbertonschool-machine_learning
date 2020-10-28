#!/usr/bin/env python3
""" doc """

import numpy as np


def pca(X, var=0.95):
    """ doc """

    U, sigma, V = np.linalg.svd(X)
    a_sum = np.cumsum(sigma)

    dim = [i for i in range(len(sigma)) if((a_sum[i]) / a_sum[-1]) >= var]
    ndim = dim[0] + 1

    return V.T[:, :ndim]
