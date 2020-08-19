#!/usr/bin/env python3
""" doc """


def moving_average(data, beta):
    """ doc """

    moving_ave = []
    V = 0
    for t, temp in enumerate(data, 1):
        V = (beta * V) + ((1 - beta) * temp)
        correction = V / (1 - (beta**t))
        moving_ave.append(correction)
    return moving_ave
