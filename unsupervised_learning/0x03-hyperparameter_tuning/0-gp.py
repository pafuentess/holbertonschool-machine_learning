#!/usr/bin/env python3
""" doc """

import numpy as np


class GaussianProcess:
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ doc """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, Y_init)

    def kernel(self, X1, X2):
        """ doc """
        factor1 = np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + factor1
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
