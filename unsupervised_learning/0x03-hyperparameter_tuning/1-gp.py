#!/usr/bin/env python3
""" doc """

import numpy as np


class GaussianProcess:
    """ doc """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ doc """
        factor1 = np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + factor1
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """ doc """
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diagonal(cov_s)

        return mu_s, sigma
