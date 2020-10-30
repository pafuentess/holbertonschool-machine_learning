#!/usr/bin/env python3
"""Function that calculates the Shannon entropy
and P affinities relative to a data point"""
import numpy as np


def HP(Di, beta):
    """Function that calculates the Shannon entropy
    and P affinities relative to a data point"""
    Pi = np.exp(-Di * beta) / np.sum(np.exp(-Di * beta))
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
