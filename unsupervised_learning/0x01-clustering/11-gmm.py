#!/usr/bin/env python3
""" doc """

import sklearn.mixture


def gmm(X, k):
    """ doc """
    Gmm = sklearn.mixture.GaussianMixture(k)
    params = Gmm.fit(X)
    clss = Gmm.predict(X)

    return (params.weights_, params.means_, params.covariances_, clss,
            Gmm.bic(X))
