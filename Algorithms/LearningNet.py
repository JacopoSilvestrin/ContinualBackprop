import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial


class LearningNet(nn.Module):

    def __init__(self, stateDim, outDim):
        super(LearningNet, self).__init__()
        hiddenLayerDim = 5
        self.l1 = nn.Linear(stateDim, hiddenLayerDim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hiddenLayerDim, outDim)

        # Initialise the weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.l2.bias)

        # Continual Backprop parameters
        self.hiddenUnits = np.zeros((hiddenLayerDim))
        self.hiddenUnitsAvg = np.zeros((hiddenLayerDim))
        self.hiddenUnitsAvgBias = np.zeros((hiddenLayerDim))
        self.hiddenUnitsAge = np.zeros((hiddenLayerDim))
        self.hiddenUtilityBias = np.zeros((hiddenLayerDim))
        self.hiddenUtility = np.zeros((hiddenLayerDim))
        self.nHiddenLayers = 1

        self.replacementRate = 1e-4
        self.decayRate = 0.99
        self.maturityThreshold = 100
        self.unitsToReplace = 0
        # List of resetted units
        self.unitReplaced = []

        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)

        self.activation = {}

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def forward(self, x):
        hook1 = self.a1.register_forward_hook(self.getActivation('h1'))
        #print(x.dtype)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        hook1.remove()

        # Update hidden units age
        self.hiddenUnitsAge += 1
        # Update hidden units estimates
        # Take hidden units values from dictionary
        self.hiddenUnits = np.reshape(self.activation['h1'].detach().numpy(),
                                            (self.hiddenUnitsAvgBias.shape[0]))
        #print("Activations: check if many are dead")
        #print(self.hiddenUnits)
        # Unbiased estimate. Warning: uses old mean estimate of the hidden units.
        self.hiddenUnitsAvg = self.hiddenUnitsAvgBias / (1 - np.power(self.decayRate, self.hiddenUnitsAge))
        # Biased estimate: updated with current hidden units values
        self.hiddenUnitsAvgBias = self.decayRate * self.hiddenUnitsAvgBias + \
                                  (1 - self.decayRate) * self.hiddenUnits

        # Compute mean-corrected contribution utility (called z in the paper)

        # Weights going out from layer l to layer l+1.
        # The i-th column of the matrix has the weights of the i-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l+i-th layer
        outgoingWeightsH1 = self.state_dict()['l2.weight'].detach().numpy()
        # Sum together contributions (sum elements of same columns) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        outgoingWeights = np.sum(np.abs(outgoingWeightsH1), axis=0).flatten()

        contribUtility = np.multiply(np.abs(self.hiddenUnits - self.hiddenUnitsAvg), outgoingWeights)

        # Compute the adaptation utility
        # Weights going in from layer l-1 to layer l.
        # The j-th row of the matrix has the weights going in the j-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l-1-th layer
        inputWeightsH1 = self.state_dict()['l1.weight'].detach().numpy()
        # Sum together contributions (sum elements of same rows) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        # The adaptation utility is the element-wise inverse of the inputWeight matrix.
        inputWeights = np.sum(np.abs(inputWeightsH1), axis=1).flatten()

        # Compute hidden unit utility
        self.hiddenUtility = self.hiddenUtilityBias / (1 - np.power(self.decayRate, self.hiddenUnitsAge))
        # Now update the hidden utility with new values
        self.hiddenUtilityBias = self.decayRate * self.hiddenUtilityBias + \
                                 (1 - self.decayRate) * contribUtility / inputWeights

        return x

    def genAndTest(self):

        nUnits = self.hiddenUnits.shape[0]


        # Select lower utility features depending on the replacement rate
        self.unitsToReplace += self.replacementRate * np.count_nonzero(self.hiddenUnitsAge > self.maturityThreshold)

        # If we accumulated enough to have one or more units to replace
        while (self.unitsToReplace >= 1):
            # Scan matrix of utilities to find lower element with age > maturityThreshold.
            min = np.amin(self.hiddenUtility)
            #minPos = 0
            minPos = []
            for i in range(self.hiddenUtility.shape[0]):
                if self.hiddenUtility[i] == min and self.hiddenUnitsAge[i] > self.maturityThreshold:
                    minPos.append(i)

            if not len(minPos):
                break

            # Pick one position randomly
            minPos = np.random.choice(minPos)
            self.unitReplaced.append(minPos)

            # Now out min and minPos values are legitimate and we can replace the input weights and set
            # to zero the outgoing weights for the selected hidden unit.
            # Set to 0 the age of the hidden unit.
            self.hiddenUnitsAge[minPos] = 0
            # Set to 0 the utilities and mean values of the hidden unit.
            self.hiddenUtilityBias[minPos] = 0
            self.hiddenUnitsAvgBias[minPos] = 0

            # Reset weights
            # Take state_dict
            weights = self.state_dict()
            # Reinitialise input weights (i-th row of previous layer)
            temp = torch.empty((1, weights['l1.weight'].shape[1]))
            torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='relu')
            weights['l1.weight'][minPos, :] = temp
            # Reset the input bias
            weights['l1.bias'][minPos] = 0
            # Set to 0 outgoing weights (i-th column of next layer) and do the same for bias
            weights['l2.weight'][:, minPos] = 0
            # weights['l2.bias'][i] = 0
            # Load stat_dict to the model to save changes
            self.load_state_dict(weights)
            # We replaced a hidden unit, reduce counter.
            self.unitsToReplace -= 1


if __name__ == "__main__":
    model = LearningNet(3, 4)
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


