#!/usr/bin/env python3
""" doc """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ doc """
    summ = 0
    for i in range(1, L + 1):
        key_dict = "W" + str(i)
        summ = summ + np.linalg.norm(weights[key_dict])
    L2cost = cost + ((lambtha / (2 * m)) * summ)
    return L2cost
