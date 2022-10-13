import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial


class TargetNet(nn.Module):

    def __init__(self, stateDim):
        super(TargetNet, self).__init__()
        hiddenLayerDim = 32
        self.l1 = nn.Linear(stateDim, hiddenLayerDim)
        #self.a1 = nn.() # TODO: Linear Activation Function (maybe I have to do it myself)
        self.l2 = nn.Linear(hiddenLayerDim, 1)

        # Initialise the weights
        with torch.no_grad():
            self.l1.weight[:,:] = torch.from_numpy(np.random.choice([-1, 1], (hiddenLayerDim, stateDim)))
            self.l1.bias[:] = torch.from_numpy(np.random.choice([-1, 1], hiddenLayerDim))
            self.l2.weight[:,:] = torch.from_numpy(np.random.choice([-1, 1], (1, hiddenLayerDim)))
            self.l2.bias[:] = torch.from_numpy(np.random.choice([-1, 1], 1))
            self.beta = 0.7

    def forward(self, x):
        m = torch.numel(x) + 1
        #print("This is the input:")
        #print(x)
        x = self.l1(x)
        #print("Input before activations:")
        #print(x)
        # Do the Linear Threshold Activation
        for i in range(x.shape[0]):
            s = 0
            if self.l1.bias.detach()[i] == -1:
                s += 1
            w_in = self.l1.weight.detach()[i,:]
            s = s + torch.numel(w_in[w_in == -1])
            theta = m * self.beta - s
            #print("Position {}".format(i))
            #print("m: {}, s: {}".format(m,s))
            #print("Theta: {}".format(theta))
            #print("x[{}] before assignment: {}".format(i,x[i]))
            if x[i] > theta:
                x[i] = 1
            else:
                x[i] = 0
            #print("x[{}] after assignment: {}".format(i, x[i]))
        #print("After activation:")
        #print(x)
        x = self.l2(x)
        #print("Output of network")
        #print(x)
        return x


if __name__ == "__main__":
    module = TargetNet(2)
    a = torch.ones(2)
    out = module(a)
    print(module.l1.weight)
    print(module.l1.bias)
    print(module.l2.weight)
    print(module.l2.bias)
    print(out)

