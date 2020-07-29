#!/usr/bin/env python3
""" doc """


class Poisson:
    """ doc """
    def __init__(self, data=None, lambtha=1.):
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        self.euler = 2.7182818285

    def factorial(self, x):
        """ doc """
        factorial = 1
        for i in range(1, x+1):
            factorial = factorial * i
        return (factorial)

    def pmf(self, k):
        """ doc """
        if k >= 0:
            k = int(k)
            num = (self.euler**(-self.lambtha) * (self.lambtha**k))
            den = self.factorial(k)
            return num / den
        else:
            return 0

    def cdf(self, k):
        """ doc """
        if k >= 0:
            k = int(k)
            pmf1 = 0
            for i in range(0, k + 1):
                pmf1 = pmf1 + self.pmf(i)
            return pmf1
        else:
            return (0)
