#!/usr/bin/env python3
""" doc """

import numpy as np


def positional_encoding(max_seq_len, dm):
    """ doc """
    encode = np.zeros([max_seq_len, dm])

    for i in range(dm):
        for p in range(max_seq_len):
            encode[p, i] = p / np.power(10000, (2 * (i // 2) / dm))

    encode[:, 0::2] = np.sin(encode[:, 0::2])
    encode[:, 1::2] = np.cos(encode[:, 1::2])

    return encode
