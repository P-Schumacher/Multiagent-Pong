import torch
import torch.nn as nn
import numpy as np
'''
This class defines a neural network model by inheritance from the torch.nn.Module class. The initializer of the parent
has to be called by super() to setup the class. Then we define our own network by nn.Sequential(). Lastly we have to
overwrite forward()
The main body of the function is debugging
'''


class DQN (nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def calc_loss(batch, net, tgt_net, GAMMA, double=False, device="cpu"):
    """
    Calculate mean squared error as loss function.
    """
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0  # final states have a future reward of 0
    next_state_values = next_state_values.detach()  # detach it from the current graph

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)


#  Main is just for debugging
if __name__ == '__main__':

    neural_net = DQN(5, 1)
    net = nn.Linear(3, 2)
    x1 = np.zeros((1, 3))
    x1_v = torch.FloatTensor(x1)
    result = net(x1_v)
    print(result)

