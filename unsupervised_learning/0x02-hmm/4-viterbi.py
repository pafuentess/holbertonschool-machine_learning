#!/usr/bin/env python3
""" doc """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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

    N = Emission.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    prev = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    prev[:, 0] = 0

    for idx, obs in enumerate(Observation):
        if idx != 0:
            F[:, idx] = np.max(F[:, idx - 1] * Transition.T *
                               Emission[np.newaxis, :, obs].T, 1)
            prev[:, idx] = np.argmax(F[:, idx - 1] * Transition.T, 1)
    path = T * [1]
    path[-1] = np.argmax(F[:, T - 1])
    for idx in reversed(range(1, T)):
        path[idx - 1] = int(prev[path[idx], idx])

    P = np.amax(F, axis=0)
    P = np.amin(P)

    return path, P
