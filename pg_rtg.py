import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym

from model import Policy
from testing import test

def train(env: gym.Env, policy: Policy, lr: float, 
          n_batches: int, batch_size: int):

    # policy optimizer
    optim = Adam(policy.parameters(), lr=lr, maximize=True)
    
    def compute_objective(obs_list, act_list, weight_list):
        obs = torch.from_numpy(np.asarray(obs_list, dtype=np.float32))
        act = torch.tensor(act_list, dtype=torch.int)
        weight = torch.tensor(weight_list, dtype=torch.float)

        logits = policy(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act)

        return (weight * log_probs).mean()

    def train_batch():

        # initialize batch
        obs_list = []          
        act_list = []           
        rtg_list = []           
        ep_len_list = []
        batch = 0

        # initialize episode
        obs, _ = env.reset()
        rew_list = []
        ep_len = 0

        # loop over time steps
        while True:

            batch += 1
            ep_len += 1

            obs_list.append(obs)

            act = policy.get_action(obs)
            act_list.append(act)

            obs, rew, term, trunc, _ = env.step(act)
            rew_list.append(rew)

            # if episode is over
            if term or trunc or batch > batch_size:
                
                # compute rtgs
                rew_list.reverse()
                rtgs = np.cumsum(rew_list).tolist()
                rtgs.reverse()

                # update lists
                rtg_list += rtgs
                ep_len_list.append(ep_len) 

                # reset episode
                obs, _ = env.reset()
                rew_list = []
                ep_len = 0

                # stop iterating if batch_size is exceeded
                if batch > batch_size:
                    break

        # compute loss and perform gradient ascent
        optim.zero_grad()
        objective = compute_objective(obs_list, act_list, rtg_list)
        objective.backward()
        optim.step()

        return objective.item(), ep_len_list 

    # main loop
    for batch in range(n_batches):
        j, ep_lens = train_batch()

        # calculate average episode length (ignore last episode)
        avg_ep_len = np.mean(ep_lens[:-1]) 
        print(f'Batch {batch:3d}: J = {j:7.2f}, '
              f'Avg Episode Len = {avg_ep_len:5.1f}')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    env = gym.make('CartPole-v1')
    policy = Policy((4, 32, 2))

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--train_eps', type=int, default=10)

    args = parser.parse_args()

    print('\nTraining CartPole\n')

    # Test
    mean, std = test(env, policy, args.train_eps)
    print(f'Before training: Avg Episode Len {mean:5.1f} +/- {std:4.1f}\n')
    
    # Train
    print('----------------------- Training -----------------------')
    train(env, policy, args.lr, args.n_batches, args.batch_size)
    
    # Test
    mean, std = test(env, policy, args.train_eps)
    print(f'\nAfter training: Avg Episode Len {mean:5.1f} +/- {std:4.1f}\n')
