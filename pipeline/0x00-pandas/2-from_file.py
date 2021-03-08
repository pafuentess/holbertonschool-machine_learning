#!/usr/bin/env python3
""" doc """

import pandas as pd


def from_file(filename, delimiter):
    """ doc """
    return pd.read_csv(filename, sep=delimiter)
