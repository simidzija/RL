"""
testing.py

Defines the `test` function, for testing a policy in a gymnasium environment.
"""

import numpy as np
import gymnasium as gym

from model import Policy

def test(env: gym.Env, policy: Policy, n_episodes: int):
    """
    Tests policy in environment. Returns average return per episode.
    """
    def test_episode():
        obs, _ = env.reset()
        rew_list = []

        # loop over time steps
        while True:
            act = policy.get_action(obs)
            obs, rew, term, trunc, _ = env.step(act)

            rew_list.append(rew)

            if term or trunc:
                return sum(rew_list)

    returns = []

    # loop over episodes (could be parallelized)
    for _ in range(n_episodes):
        ret = test_episode()
        returns.append(ret)

    mean = np.mean(returns)
    std = np.std(returns)
            
    return mean, std