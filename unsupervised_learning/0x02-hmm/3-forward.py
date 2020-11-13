#!/usr/bin/env python3
""" doc """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ doc """
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), [1])[0]:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if not np.isclose(np.sum(Transition, axis=1),
                      np.ones(Initial.shape[0])).all():
        return None, None
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None
    T = Observation.shape[0]
    N = Initial.shape[0]
    F = np.zeros((N, T))
    idx = Observation[0]
    idx_emission = Emission[:, idx]
    F[:, 0] = Initial.T * idx_emission

    for i in range(1, T):
        for j in range(N):
            F[j, i] = np.sum(Emission[j,
                             Observation[i]] * Transition[:, j] * F[:, j - 1],
                             axis=0)
    P = np.sum(F[:, T - 1:], axis=0)[0]

    return P, F
