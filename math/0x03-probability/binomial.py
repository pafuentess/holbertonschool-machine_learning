#!/usr/bin/env python3
""" doc """


class Binomial:
    """ doc """

    def __init__(self, data=None, n=1, p=0.5):
        """ doc """
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            pre_var = []
            for i in data:
                pre_var.append(float((i - self.mean) ** 2))
            var = float(sum(pre_var) / (len(pre_var))
            self.p = float(1 - (var / self.mean))
            self.n = round(self.mean / self.p)
            self.p = float(self.mean /self.n)
            """
            pre_var = []
            for i in data:
                pre_var.append(float((i - self.mean) ** 2))
            var = float(sum(pre_var) / len(pre_var))
            """
        else:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

    def factorial(self, x):
        """ doc """
        factorial = 1
        for i in range(1, x+1):
            factorial = factorial * i
        return (factorial)

    def pmf(self, k):
        """ doc """
        if k <= self.n:
            k = int(k)
            n_fact = self.factorial(self.n)
            k_fact = self.factorial(k)
            nfact_kfact = self.factorial(self.n - k)
            coef1 = n_fact / (k_fact * nfact_kfact)
            coef2 = (1 - self.p) ** (self.n - k)
            return coef1 * (self.p ** k) * coef2
        else:
            return 0

    def cdf(self, k):
        """ doc """
        if k <= self.n:
            k = int(k)
            acul = 0
            for i in range(0, k + 1):
                acul = acul + self.pmf(i)
            return acul
