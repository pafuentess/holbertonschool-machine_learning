#!/usr/bin/env python3
""" doc """
import numpy as np


class MultiNormal:
    """ doc """
    def __init__(self, data):
        """Class contructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.mean = mean
        self.cov = np.dot(data - self.mean, data.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """ doc """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        constant = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        neg_dev = -(x - self.mean).T

        inner = np.matmul(neg_dev, inv)
        half_dev = (x - self.mean) / 2
        outer = np.matmul(inner, half_dev)
        f = np.exp(outer)
        pdf = constant * f

        return pdf.reshape(-1)[0]
