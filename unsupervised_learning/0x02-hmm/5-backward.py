#!/usr/bin/env python3
""" doc """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
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
    B = np.ones((N, T))

    for obs in reversed(range(T-1)):
        for h_state in range(N):
            B[h_state, obs] = (np.sum(B[:, obs + 1] *
                               Transition[h_state, :] *
                               Emission[:, Observation[obs + 1]]))

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
