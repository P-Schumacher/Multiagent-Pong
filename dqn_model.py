import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

'''
This class defines a neural network model by inheritance from the torch.nn.Module class. The initialiser of the parent 
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

if __name__ == '__main__':

    neural_net = DQN(5, 1)
    net = nn.Linear(3, 2)
    x1 = np.zeros((1, 3))
    x1_v = torch.FloatTensor(x1)
    result = net(x1_v)
    print(result)

