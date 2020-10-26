#!/usr/bin/env python3

import numpy as np


class MultiNormal:
    """ doc """
    def __init__(self, data):
        """ doc """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            err = 'data must be a 2D numpy.ndarray'
            raise TypeError(err)
        d, n = data.shape

        if n < 2:
            err = 'data must contain multiple data points'
            raise ValueError(err)

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        deviaton = data - self.mean
        self.cov = np.matmul(deviaton, deviaton.T) / (n - 1)

    def pdf(self, x):
        """ doc """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        constant = 1 / np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(self.cov)))
        neg_dev = -(x - self.mean).T

        inner = np.matmul(neg_dev,  np.linalg.inv(self.cov))
        half_dev = (x - self.mean) / 2
        outer = np.matmul(inner, half_dev)
        f = np.exp(outer)
        pdf = constant * f
        pdf = pdf.reshape(-1)[0]

        return pdf
