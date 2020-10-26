#!/usr/bin/env python3
""" contain the multinormal class"""

import numpy as np


class MultiNormal:
    """
    Multinormal class
    """

    def __init__(self, data):
        """
        constructor
        :param data: numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
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
        """
        calculates the PDF at a data point
        :param x: numpy.ndarray of shape (d, 1)
            containing the data point whose PDF should be calculated
        :return:  value of the PDF
        """
        if not isinstance(x, np.ndarray):
            err = 'x must be a numpy.ndarray'
            raise TypeError(err)

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