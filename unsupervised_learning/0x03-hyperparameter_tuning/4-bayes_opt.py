#!/usr/bin/env python3
""" doc """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ doc """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ doc """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ doc """
        X = self.gp.X
        mu_sample, _ = self.gp.predict(X)
        mu, sigma = self.gp.predict(self.X_s)

        sigma = sigma.reshape(-1, 1)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            if self.minimize is True:
                mu_sample_opt = np.amin(self.gp.Y)
                imp = (mu_sample_opt - mu - self.xsi).reshape(-1, 1)
            else:
                mu_sample_opt = np.amax(self.gp.Y)
                imp = (mu - mu_sample_opt - self.xsi).reshape(-1, 1)

            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        Next = self.X_s[np.argmax(ei)]

        return Next, ei.reshape(-1)
