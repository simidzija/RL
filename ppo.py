"""
ppo.py

--------------------------------------------------------------------------------
                        Proximal Policy Optimization
--------------------------------------------------------------------------------

Implementation of the proximal policy optimization (PPO) algorithm, including 
generalized advantage estimation (GAE) for the advantage function. This 
implementation is compatible with RL enviornments provided by the `gymnasium` 
library.

We test the algorithm on the `CartPole-v1` environment. After 10 independent 
training runs the algorithm achieves a mean score of 478 with a standard 
deviation of 24. It commonly achieves a perfect score of 500 in less than 50 
epochs of training. From experimention the algorithm appears to be more robust 
to the choice of hyperparameters as compared to VPG.
"""

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.nn import MSELoss
import gymnasium as gym

from model import Policy, Value
from testing import test

def vpg(
        env: gym.Env, 
        policy: Policy, 
        value: Value, 
        lr_pol: float, 
        lr_val: float, 
        n_batches: int, 
        batch_size: int,
        pol_train_steps: int,
        val_train_steps: int,
        epsilon: float,
        gamma: float, 
        lambd: float,
        kl_cutoff: float,
        completion_reward: float
):

    # optimizers
    opt_pol = Adam(policy.parameters(), lr=lr_pol, maximize=True)
    opt_val = Adam(value.parameters(), lr=lr_val)
    
    # objective functions
    def compute_obj_pol(prob_old: Tensor, prob_new: Tensor, adv: Tensor):
        ratio = prob_new / prob_old
        unclipped = ratio * adv
        clipped = torch.clip(ratio, 1 - epsilon, 1 + epsilon) * adv

        return torch.minimum(unclipped, clipped).mean()
    
    def compute_obj_val(val, rtg):
        return MSELoss()(val, rtg)
    
    # GAE matrix containing powers of x = gamma * lambd
    def construct_gae_matrix():
        max_steps = env.spec.max_episode_steps
        x = gamma * lambd
        # create vector (1, x, x^2, ...)
        gae_vector = np.array([x**k for k in range(max_steps)])
        # create matrix of zeros
        gae_matrix = np.zeros((max_steps, max_steps))
        # populate non-zero elements of matrix
        for i, row in enumerate(gae_matrix):
            row[i:] = gae_vector[:max_steps - i]
        return gae_matrix
    
    gae_matrix = construct_gae_matrix()    

    def train_batch():

        # initialize batch
        obs_list = []
        act_list = []
        rtg_list = []
        adv_list = []
        ep_len_list = []
        batch = 0

        # initialize episode
        obs, info = env.reset()
        rew_list = []
        ep_len = 0

        # loop over time steps
        while True:

            batch += 1
            ep_len += 1

            # observation
            obs_list.append(obs)

            # action
            act = policy.get_action(obs)
            obs, rew, term, trunc, info = env.step(act)
            act_list.append(act)

            # reward
            rew_list.append(rew)

            # if episode is over
            # if term or trunc or batch > batch_size:
            if term or trunc:
                
                # compute rtgs
                rew_list.reverse()
                rtgs = np.cumsum(rew_list).tolist()
                rtgs.reverse()

                # reward agent who makes it to the end
                if ep_len == env.spec.max_episode_steps:
                    rtgs = [r + completion_reward for r in rtgs]

                # compute values
                val_list = value.get_value(obs_list + [obs])

                # compute temporal difference (TD) residuals
                td_array = np.array([r + gamma * v_next - v for r, v, v_next 
                           in zip(rew_list, val_list[:-1], val_list[1:])])
                
                # compute advantages
                adv_array = gae_matrix[:ep_len, :ep_len] @ td_array

                # update lists
                rtg_list += rtgs
                adv_list += adv_array.tolist()
                ep_len_list.append(ep_len) 

                # reset episode
                obs, info = env.reset()
                rew_list = []
                ep_len = 0

                # stop iterating if batch_size is exceeded
                if batch > batch_size:
                    break

        # batch tensors
        obs_ten = torch.from_numpy(np.asarray(obs_list, dtype=np.float32))
        act_ten = torch.tensor(act_list, dtype=torch.int64)
        rtg_ten = torch.tensor(rtg_list, dtype=torch.float)
        adv_ten = torch.tensor(adv_list, dtype=torch.float)

        # policy gradient ascent
        prob_old = policy.get_prob(obs_ten, act_ten, compute_grads=False)
        for i in range(pol_train_steps):
            opt_pol.zero_grad()
            prob_new = policy.get_prob(obs_ten, act_ten)
            obj_pol = compute_obj_pol(prob_old, prob_new, adv_ten)
            obj_pol.backward()
            opt_pol.step()
            # break if KL div too large
            kldiv = (prob_old * (prob_old / prob_new).log()).mean()
            if kldiv > kl_cutoff:
                break

        # value function gradient descent
        for i in range(val_train_steps):
            opt_val.zero_grad()
            val_ten = value(obs_ten).squeeze()
            obj_val = compute_obj_val(val_ten, rtg_ten)
            obj_val.backward()
            opt_val.step()

        return obj_pol.item(), obj_val.item(), ep_len_list 

    # main loop
    for batch in range(n_batches):
        obj_pol, obj_val, ep_lens = train_batch()

        # calculate average episode length (ignore last episode)
        avg_ep_len = np.mean(ep_lens[:-1]) 
        print(f'Batch {batch:3d}: '
              f'obj_pol = {obj_pol:7.2f}, obj_val = {obj_val:7.2f}, '
              f'Avg Episode Len = {avg_ep_len:5.1f}')
        
    env.close()
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    env = gym.make('CartPole-v1')
    policy = Policy((4, 64, 2))
    value = Value((4, 64, 1))

    parser.add_argument('--lr_pol', type=float, default=1e-3)
    parser.add_argument('--lr_val', type=float, default=1e-3)
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--pol_train_steps', type=int, default=10)
    parser.add_argument('--val_train_steps', type=int, default=10)
    parser.add_argument('--epsilon', type=int, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--kl_cutoff', type=float, default=0.02)
    parser.add_argument('--train_eps', type=int, default=10)
    parser.add_argument('--completion_reward', type=float, default=0.)

    args = parser.parse_args()

    print('\nTraining CartPole\n')

    # Test
    mean, std = test(env, policy, args.train_eps)
    print(f'Before training: Avg Episode Len {mean:5.1f} +/- {std:4.1f}\n')
    
    # Train
    print('----------------------- Training -----------------------')
    vpg(env, 
        policy, 
        value, 
        lr_pol = args.lr_pol, 
        lr_val = args.lr_val, 
        n_batches = args.n_batches, 
        batch_size = args.batch_size,
        pol_train_steps = args.pol_train_steps,
        val_train_steps = args.val_train_steps, 
        epsilon = args.epsilon,
        gamma = args.gamma, 
        lambd = args.lambd,
        kl_cutoff = args.kl_cutoff,
        completion_reward = args.completion_reward
    )
    
    # Test
    mean, std = test(env, policy, args.train_eps)
    print(f'\nAfter training: Avg Episode Len {mean:5.1f} +/- {std:4.1f}\n')
