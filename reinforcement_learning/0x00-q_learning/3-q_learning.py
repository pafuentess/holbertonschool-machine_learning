#!/usr/bin/env python3

import numpy as np
epsilonGreedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ performs Q-learning:
        - env is the FrozenLakeEnv instance
        - Q is a numpy.ndarray containing the Q-table
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
        - epsilon is the initial threshold for epsilon greedy
        - min_epsilon is the minimum value that epsilon should decay to
        - epsilon_decay is the decay rate for updating epsilon between episodes
        - When the agent falls in a hole, the reward should be updated to be -1
        - Returns: Q, total_rewards
                   Q is the updated Q-table
                   total_rewards is a list containing the rewards per episode
    """
    rewardsAllEpi = []
    initialEps = epsilon

    for episode in range(episodes):
        state = env.reset()
        rewardSum = 0

        for step in range(max_steps):
            action = epsilonGreedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            state = new_state
            rewardSum += reward

            if done:
                break

        rewardsAllEpi.append(rewardSum)

        epsilon = (min_epsilon + (initialEps - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return (Q, rewardsAllEpi)
