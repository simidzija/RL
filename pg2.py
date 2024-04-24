import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn import MSELoss
import gymnasium as gym

from model import Policy, Value
from testing import test

def pg2(env: gym.Env, policy: Policy, value: Value, 
          lr_pol: float, lr_val: float, n_batches: int, batch_size: int):

    # optimizers (j = policy objective, v = value function)
    opt_pol = Adam(policy.parameters(), lr=lr_pol, maximize=True)
    opt_val = Adam(value.parameters(), lr=lr_val)
    
    def compute_policy_objective(obs, act, weight):
        logits = policy(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act)

        return (weight * log_probs).mean()
    
    def compute_value_objective(val, rtg):
        loss_fn = MSELoss()
        return loss_fn(val, rtg)

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

            # observation
            obs_list.append(obs)

            # action
            act = policy.get_action(obs)
            obs, rew, term, trunc, _ = env.step(act)
            act_list.append(act)

            # reward
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

        # batch tensors
        obs_ten = torch.from_numpy(np.asarray(obs_list, dtype=np.float32))
        act_ten = torch.as_tensor(act_list, dtype=torch.int)
        rtg_ten = torch.as_tensor(rtg_list, dtype=torch.float)
        val_ten = value(obs_ten).squeeze()
        weight_ten = torch.as_tensor([r - v for r, v in zip(rtg_ten, val_ten)])

        # policy gradient ascent
        opt_pol.zero_grad()
        obj_pol = compute_policy_objective(obs_ten, act_ten, weight_ten)
        obj_pol.backward()
        opt_pol.step()

        # value function gradient descent
        opt_val.zero_grad()
        obj_val = compute_value_objective(val_ten, rtg_ten)
        obj_val.backward()
        opt_val.step()

        return obj_pol.item(), obj_val.item(), ep_len_list 

    # main loop
    for batch in range(n_batches):
        obj_pol, obj_val, ep_lens = train_batch()

        # calculate average episode length (ignore last episode)
        avg_ep_len = np.mean(ep_lens[:-1]) 
        print(f'Batch {batch:3d}: '
              f'obj_pol = {obj_pol:7.2f}, obj_v = {obj_val:7.2f}, '
              f'Avg Episode Len = {avg_ep_len:5.1f}')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    env = gym.make('CartPole-v1')
    policy = Policy((4, 32, 2))
    value = Value((4,32,1))

    parser.add_argument('--lr_pol', type=float, default=0.01)
    parser.add_argument('--lr_val', type=float, default=0.01)
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
    pg2(env, policy, value, args.lr_pol, args.lr_val, 
          args.n_batches, args.batch_size)
    
    # Test
    mean, std = test(env, policy, args.train_eps)
    print(f'\nAfter training: Avg Episode Len {mean:5.1f} +/- {std:4.1f}\n')

