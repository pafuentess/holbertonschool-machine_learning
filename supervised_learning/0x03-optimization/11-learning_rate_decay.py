#!/usr/bin/env python3
""" doc """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ doc """
    decay = alpha / (1 + decay_rate * (int(global_step / decay_step)))
    return decay
