#!/usr/bin/env python3
""" doc """


class Normal:
    """ doc """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ doc """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(sum(data) / len(data))
            pre_var = []
            for i in data:
                pre_var.append(float((i - self.mean) ** 2))
            var = float(sum(pre_var) / len(pre_var))
            self.stddev = float(var ** (0.5))
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        self.euler = 2.7182818285
        self.pi = 3.1415926536

    def z_score(self, x):
        """ doc """
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """ doc """
        return ((z * self.stddev) + self.mean)

    def errF(self, x):
        """ doc """
        coef1 = float(2 / ((self.pi) ** (0.5)))
        coef2 = float(x - ((x ** 3) / 3) + ((x ** 5) / 10) -
                      ((x ** 7) / 42) + ((x ** 9) / 216))
        return coef1 * coef2

    def pdf(self, x):
        """ doc """
        coef1 = float((1 / ((2 * self.pi * (self.stddev) ** 2) ** (0.5))))
        coef2 = float(((x - self.mean) ** 2 / (2 * (self.stddev) ** 2)))
        coef3 = float(self.euler ** (-coef2))
        return float(coef1 * coef3)

    def cdf(self, x):
        """ doc """
        coefx = (x - self.mean) / (self.stddev * (2 ** (0.5)))
        err = self.errF(coefx)
        return 0.5 * (1 + err)
