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
            a = Emission[j, Observation[i]]
            b = Transition[:, j]
            c = F[:, i - 1]
            F[j, i] = np.sum(a * b * c, axis=0)
    P = np.sum(F[:, T - 1:], axis=0)[0]
    return P, F


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
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
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None

    N = Initial.shape[0]
    T = Observations.shape[0]
    M = Emission.shape[1]

    a = Transition
    b = Emission
    a_prev = np.copy(a)
    b_prev = np.copy(b)

    for iteration in range(1000):
        PF, F = forward(Observations, b, a, Initial)
        PB, B = backward(Observations, b, a, Initial)
        X = np.zeros((N, N, T - 1))
        NUM = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    Fit = F[i, t]
                    aij = a[i, j]
                    bjt1 = b[j, Observations[t + 1]]
                    Bjt1 = B[j, t + 1]
                    NUM[i, j, t] = Fit * aij * bjt1 * Bjt1
        DEN = np.sum(NUM, axis=(0, 1))
        X = NUM / DEN
        G = np.zeros((N, T))
        NUM = np.zeros((N, T))
        for t in range(T):
            for i in range(N):
                Fit = F[i, t]
                Bit = B[i, t]
                NUM[i, t] = Fit * Bit
        DEN = np.sum(NUM, axis=0)
        G = NUM / DEN
        a = np.sum(X, axis=2) / np.sum(G[:, :T - 1], axis=1)[..., np.newaxis]
        DEN = np.sum(G, axis=1)
        NUM = np.zeros((N, M))
        for k in range(M):
            NUM[:, k] = np.sum(G[:, Observations == k], axis=1)
        b = NUM / DEN[..., np.newaxis]
        if np.all(np.isclose(a, a_prev)) or np.all(np.isclose(a, a_prev)):
            return a, b
        a_prev = np.copy(a)
        b_prev = np.copy(b)
    return a, b
