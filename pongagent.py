'''
Module pongagent

FUNCTIONS
---------
start_env(params)
fill_buffer(agent, start_size, replay_size)
load_model(net, tgt_net, filename)
_setup_all(params)
_construct_env(env, n_actions, n_skip)

'''
import helpfunc
import helpclass
from buffers import ExperienceBuffer
import torch
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter

class Pongagent:
    def __init__(self, env, exp_buffer:ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.exp_buffer.fill = lambda nsamples: self.exp_buffer.fill(self.env, nsamples)
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        Epsilon greedy step. With probability epsilon, a random action is taken (exploration),
        else the action ist chosen to maximize the q-value as approximated by net (exploitation).
        """
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.FloatTensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def train(params):
    pass