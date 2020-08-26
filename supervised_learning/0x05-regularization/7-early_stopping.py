#!/usr/bin/env python3
""" doc """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ doc """
    check = True
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count != patience:
        check = False
    return check, count
