#!/usr/bin/env python3
""" epsilon-greedy """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ uses epsilon-greedy to determine the next action:
        - Q is a numpy.ndarray containing the q-table
        - state is the current state
        - epsilon is the epsilon to use for the calculation
        - You should sample p with numpy.random.uniformn to determine
          if your algorithm should explore or exploit
        - If exploring, you should pick the next action with
          numpy.random.randint from all possible actions
        - Returns: the next action index
    """

    action = Q.shape[1]

    if np.random.uniform(0, 1) < epsilon:
        A_index = np.random.randint(action)
    else:
        A_index = np.argmax(Q[state])

    return A_index
