#!/usr/bin/env python3
""" doc """


class Exponential:
    """ doc """
    def __init__(self, data=None, lambtha=1.):
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(len(data) / sum(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)

        self.euler = 2.7182818285

    def pdf(self, x):
        """ doc """
        if x >= 0:
            return self.lambtha * (self.euler**(-self.lambtha * x))
        else:
            return 0

    def cdf(self, x):
        """ doc """
        if x >= 0:
            return 1 - (self.euler**(-self.lambtha * x))
        else:
            return 0
