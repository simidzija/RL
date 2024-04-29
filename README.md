# RL

Implementation of various policy based reinforcement learning algorithms, 
detailed in Open AI's [Spinning Up](https://spinningup.openai.com/) resource.
In order of complexity, these include:
- [pg0.py](pg0.py): Use full return to estimate policy grad.
- [pg1.py](pg1.py): Use reward-to-go to estimate policy grad.
- [pg2.py](pg2.py): Use reward-to-go and value baseline.
- [vpg.py](vpg.py): Vanilla Policy Gradient.
- [ppo.py](ppo.py): Proximal Policy Optimization.

The implementations are compatible with RL environments provided by the 
[`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) library. 
We test the algorithms on the CartPole-v1 environment.
