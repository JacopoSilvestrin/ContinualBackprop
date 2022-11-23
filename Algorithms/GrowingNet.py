import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial


class GrowingNet(nn.Module):

    def __init__(self, stateDim, outDim):
        super(GrowingNet, self).__init__()
        hiddenLayerDim = 5
        self.l1 = nn.Linear(stateDim, hiddenLayerDim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hiddenLayerDim, outDim)

        self.I = stateDim
        self.H = hiddenLayerDim
        self.O = outDim

        # Initialise the weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        '''print("Before initialazing bias")
        print(self.l1.bias.data)
        torch.nn.init.zeros_(self.l1.bias)
        print("Post")
        print(self.l1.bias.data)'''
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.l2.bias)

        self.additionPeriod = 1e4
        self.counter = 0

        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        return x

    def growNet(self, no_of_neurons):

        if self.counter % self.additionPeriod != 0 or self.counter == 0:
            self.counter += 1
            return
        with torch.no_grad():
            weights = [self.l1.weight.data, self.l2.weight.data]
            biases = [self.l1.bias.data, self.l2.bias.data]
            self.l1 = torch.nn.Linear(self.I, self.H + no_of_neurons)
            self.l2 = torch.nn.Linear(self.H + no_of_neurons, self.O)

            self.l1.weight.data[0:-no_of_neurons, :] = weights[0]
            temp = torch.empty((no_of_neurons, self.l1.weight.data.shape[1]))
            torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='relu')
            self.l1.weight[-no_of_neurons:,:] = temp
            self.l1.bias.data[0:-no_of_neurons] = biases[0]
            self.l1.bias.data[-no_of_neurons:] = 0

            self.l2.weight.data[:, 0:-no_of_neurons] = weights[1]
            temp = torch.empty((self.l2.weight.data.shape[0], no_of_neurons))
            torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='relu')
            self.l2.weight[:, -no_of_neurons:] = temp
            self.l2.bias.data[:] = biases[1]

            self.H = self.H + no_of_neurons
            self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)

        '''print("Check weights l1:")
        print(self.l1.weight.data)
        print("Check bias l1:")
        print(self.l1.bias.data)
        print("Check weights l2:")
        print(self.l2.weight.data)
        print("Check bias l2:")
        print(self.l2.bias.data)'''


        self.counter += 1
        return



if __name__ == "__main__":
    model = GrowingNet(3, 4)
    input = torch.tensor([1., 2., 3.])

    # print(model.parameters())

    for param in model.parameters():
        # print("Start printing parameters:")
        print("param before setting to 1:")
        print(param.data)
        param.data = nn.parameter.Parameter(torch.ones_like(param))
        print("Param after setting to 1:")
        print(param.data)

    # Take state_dict
    inWeights = model.state_dict()
    print("Old layer 1:")
    print(inWeights['l1.weight'])
    # Reinitialise input weights (i-th row of previous layer)
    temp = torch.empty((1, inWeights['l1.weight'].size()[1]))
    torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='tanh')
    inWeights['l1.weight'][0, :] = temp
    print("New layer 1:")
    print(inWeights['l1.weight'])

    print("Old layer 2:")
    print(inWeights['l2.weight'])
    print(inWeights['l2.bias'])
    # Set to 0 outgoing weights (i-th column of next layer)
    inWeights['l2.weight'][:, 0] = 0
    inWeights['l2.bias'][0] = 0
    print("New layer 2:")
    print(inWeights['l2.weight'])
    print(inWeights['l2.bias'])
    # Load stat_dict to the model to save changes
    model.load_state_dict(inWeights)
    print("Now you should see all the modifications:")
    for param in model.parameters():
        # print("Start printing parameters:")
        print(param.data)

    '''
    print("Weights and biases are set to 1, let's forward and see the output:")
    out = model.forward(input)
    print("Output:")
    print(out)
    print("Activations:")
    print(model.activation)
    w1 = np.ones((3,3))
    bias = np.ones((3))
    i1 = np.matmul(np.array([1.,2.,3.]), w1)
    print("bias:")
    print(bias)
    i1 = i1 + bias
    i1 = np.tanh(i1)
    print("Prova layer 1:")
    print(i1)
    i2 = np.matmul(i1,w1) + bias
    i2 = np.tanh(i2)
    print("i2:")
    print(i2)

    wout = np.ones((3,4))
    out_final = np.matmul(i2, wout) + np.ones((4))
    print("Out:")
    print(out_final)

    print("Check type activations:")
    print(model.activation['h1'].detach().numpy().dtype)
    '''

    # print(model.state_dict()['l2.weight '])


