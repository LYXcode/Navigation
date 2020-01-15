import torch
from collections import namedtuple
import gym

env = gym.make('CartPole-v0')
print(env.x_threshold)
print(env.y_threshold)
# x = torch.ones(2, 2)
# print(x)
# print(x.view(x.size(0), -1))