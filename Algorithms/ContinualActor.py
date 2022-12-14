import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial

class Actor(nn.Module):

    def __init__(self, stateDim, actionDim, mode): # mode can be either fisher or cbp
        super(Actor, self).__init__()
        self.I = stateDim
        self.H = 32
        self.O = actionDim
        self.l1 = nn.Linear(self.I, self.H)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(self.H, self.H)
        self.a2 = nn.Tanh()
        self.l3 = nn.Linear(self.H, self.O)
        #self.model = nn.Sequential(self.l1, self.a1, self.l2, self.a2, self.l3)
        self.layerList = nn.ModuleList(self.l1, self.l2, self.l3)
        self.activationList = nn.ModuleList(self.a1, self.a2)

        # Initialise the weights
        #torch.nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='tanh')
        #torch.nn.init.zeros_(self.l1.bias)
        #torch.nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='tanh')
        #torch.nn.init.zeros_(self.l2.bias)
        #torch.nn.init.kaiming_uniform_(self.l3.weight, mode='fan_in', nonlinearity='tanh')
        #torch.nn.init.zeros_(self.l3.bias)

        # Get operating mode
        self.mode = mode
        if mode != "cbp" and mode != "fisher" and mode != "debug":
            print("Warning: unknown mode.")

        # Get number of hidden layers and number of parameters
        self.nParams = 0
        self.nLayers = 0
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                self.nLayers += 1
                self.nParams += torch.numel(layer.weight)
                self.nParams += torch.numel(layer.bias)
        self.nHiddenLayers = self.nLayers - 1

        # Continual Backprop parameters
        self.hiddenUnits = torch.zeros((self.H, self.nHiddenLayers))
        self.hiddenUnitsAvg = torch.zeros((self.H, self.nHiddenLayers))
        self.hiddenUnitsAvgBias = torch.zeros((self.H, self.nHiddenLayers))
        self.hiddenUnitsAge = np.zeros((self.H, self.nHiddenLayers)) # Age is numpy
        self.hiddenUnitsCount = torch.zeros((self.H, self.nHiddenLayers))
        self.hiddenUtilityBias = torch.zeros((self.H, self.nHiddenLayers))
        self.hiddenUtility = torch.zeros((self.H, self.nHiddenLayers))

        # Fisher Backprop parameters
        self.hiddenFisherUnitsAge = np.zeros((self.H, self.nHiddenLayers))
        #self.hiddenFisherUnitsCount = torch.zeros((self.H, self.nHiddenLayers))
        self.F = torch.zeros((self.nParams, self.nParams))
        self.FCount = 0
        self.fisherUtility = torch.zeros((self.H, self.nHiddenLayers))

        # Reset parameters
        self.replacementRate = 10e-4
        self.decayRate = 0.99
        self.maturityThreshold = 100
        self.unitsToReplace = np.zeros(self.nHiddenLayers)

        # Growing net params
        self.counter = 0
        self.growPeriod = 1e4

        self.optimizer = torch.optim.SGD(self.parameters(), lr=10e-4)

        self.activation = {}

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def isActivation(self, module):
        if isinstance(module, nn.Tanh) or isinstance(module, nn.ReLU):
            return True
        return False

    def forward(self, x):
        # Set up hooks on activations
        hooks = []
        hookID = 1
        for module in self.activationList:
            if self.isActivation(module):
                hooks.append(self.module.register_forward_hook(self.getActivation("h{}".format(hookID))))
                hookID += 1
            else:
                print("Not an activation, something is wrong")
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        # Remove hooks
        for hook in hooks:
            hook.remove()

        if self.mode == "cbp" or self.mode == "debug":
            self.updateCBPUtility()
        return x

    def updateCBPUtility(self): # To be called during forward
        # Update count
        self.hiddenUnitsCount += 1
        # Update hidden units estimates
        # Take hidden units values from dictionary
        for i in range(self.nHiddenLayers):
            self.hiddenUnits[:, i] = torch.reshape(self.activation['h{}'.format(i)].detach(),
                                                   (self.hiddenUnits.shape[0]))
        # Unbiased estimate. Warning: uses old mean estimate of the hidden units.
        self.hiddenUnitsAvg = self.hiddenUnitsAvgBias / (1 - torch.pow(self.decayRate, self.hiddenUnitsCount))
        # Biased estimate: updated with current hidden units values
        self.hiddenUnitsAvgBias = self.decayRate * self.hiddenUnitsAvgBias + \
                                  (1 - self.decayRate) * self.hiddenUnits

        # Compute mean-corrected contribution utility (called z in the paper)
        # Get weights
        weights = []
        for layer in self.layerList:
            if isinstance(layer, nn.Linear):
                layerWeights = layer.weight.detach()
                # Sum together contributions (sum elements of same columns) from same hidden unit
                # and reshape to obtain h1xh2 matrix to use in the formula.
                weights.append(layerWeights)

        # Weights going out from layer l to layer l+1.
        # The i-th column of the matrix has the weights of the i-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l+i-th layer
        outgoingWeights = torch.zeros((self.H, self.nHiddenLayers))
        inputWeights = torch.zeros((self.H, self.nHiddenLayers))
        for i in range(len(weights)):
            # Discard first layer as we want weights coming out from hidden features
            if i != 0:
                # Sum together contributions (sum elements of same columns) from same hidden unit
                # and reshape to obtain h1xh2 matrix to use in the formula.
                outgoingWeights[:, i - 1] = torch.reshape(torch.sum(torch.abs(weights[i]), dim=0), (-1, 1))
            if i != len(weights) - 1:
                inputWeights[:, i] = torch.reshape(torch.sum(torch.abs(weights[i]), dim=1), (-1, 1))

        contribUtility = torch.mul(torch.abs(self.hiddenUnits - self.hiddenUnitsAvg), outgoingWeights)

        # Compute the adaptation utility (did it above)
        # Weights going in from layer l-1 to layer l.
        # The j-th row of the matrix has the weights going in the j-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l-1-th layer
        # inputWeightsH1 = self.state_dict()['l1.weight'].detach().numpy()
        # inputWeightsH2 = self.state_dict()['l2.weight'].detach().numpy()

        # Sum together contributions (sum elements of same rows) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        # The adaptation utility is the element-wise inverse of the inputWeight matrix.
        # inputWeights = np.hstack((np.reshape(np.sum(np.abs(inputWeightsH1), axis=1), (-1, 1)),
        #                          np.reshape(np.sum(np.abs(inputWeightsH2), axis=1), (-1, 1))))

        # Compute hidden unit utility
        self.hiddenUtility = self.hiddenUtilityBias / (1 - torch.pow(self.decayRate, self.hiddenUnitsCount))
        # Now update the hidden utility with new values
        self.hiddenUtilityBias = self.decayRate * self.hiddenUtilityBias + \
                                 (1 - self.decayRate) * contribUtility / inputWeights

    def CBPReset(self, loss):

        self.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update hidden units age
        self.hiddenUnitsAge += 1
        nUnits = self.hiddenUnits.shape[0]

        weights = []
        biases = []
        for layer in self.layerList:
            if isinstance(layer, nn.Linear):
                layerWeights = layer.weight.detach()
                layerBias = layer.bias.detach()
                # Sum together contributions (sum elements of same columns) from same hidden unit
                # and reshape to obtain h1xh2 matrix to use in the formula.
                weights.append(layerWeights)
                biases.append(layerBias)

        # Do the same for each layer.
        for j in range(self.nHiddenLayers):
            # Select lower utility features depending on the replacement rate
            self.unitsToReplace[j] += self.replacementRate * np.count_nonzero(self.hiddenUnitsAge > self.maturityThreshold)

            while(self.unitsToReplace[j] >= 1):
                # Scan matrix of utilities to find lower element with age > maturityThreshold.
                min = torch.amin(self.hiddenUtility[:, j])
                minPos = []
                for i in range(self.hiddenUtility.shape[0]):
                    if self.hiddenUtility[i, j] == min and self.hiddenUnitsAge[i, j] > self.maturityThreshold:
                        minPos.append(i)

                if not len(minPos):
                    break

                # Pick one position randomly
                minPos = np.random.choice(minPos)

                # Now out min and minPos values are legitimate and we can replace the input weights and set
                # to zero the outgoing weights for the selected hidden unit.
                # Set to 0 the age of the hidden unit.
                self.hiddenUnitsAge[minPos, j] = 0
                self.hiddenUnitsCount[minPos, j] = 0
                # Set to 0 the utilities and mean values of the hidden unit.
                self.hiddenUtilityBias[minPos, j] = 0
                self.hiddenUnitsAvgBias[minPos, j] = 0

                # Reset weights
                # Input weights
                temp = nn.Linear(self.layerList[j].weight.shape[1], self.layerList[j].weight.shape[0])
                self.layerList[j].weight[minPos,:] = temp.weight.detach()[0, :]
                self.layerList[j].bias[minPos] = 0
                self.layerList[j+1].weight[:, minPos] = 0

                # We replaced a hidden unit, reduce counter.
                self.unitsToReplace -= 1

    def computeFisher(self):
        # Computation of Fisher
        self.hiddenFisherUnitsAge += 1
        paramsVec = torch.tensor([])
        for layer in self.layerList:
            if isinstance(layer, nn.Linear):
                weightsGrad = layer.weight.grad.detach().flatten()
                biasGrad = layer.bias.grad.detach().flatten()
                paramsVec = torch.concatenate((paramsVec, weightsGrad, biasGrad), dim=0)

        self.F = self.decayRate * self.F + (1 - self.decayRate) * torch.outer(paramsVec, paramsVec)

    def computeFUtility(self):
        # Compute Fisher utility summing contrib for input and output layers of each hidden unit
        # Take weight and bias for each layer
        params = np.empty(self.nLayers, dtype=object)
        count = 0
        for layer in self.layerList:
            if isinstance(layer, nn.Linear):
                params[count] = [layer.weight.data.detach(), layer.bias.data.detach()]
                count += 1
        # Compute utility
        for j in range(self.nHiddenLayers):
            input_w = params[j][0]
            input_b = params[j][1]
            output_w = params[j + 1][0]
            output_b = params[j + 1][1]
            for i in range(self.H):
                i_w = torch.zeros_like(input_w)
                i_w[i, :] = -1 * input_w[i, :]
                i_b = torch.zeros_like(input_b)
                i_b[i] = -1 * input_b[i]
                o_w = torch.zeros_like(output_w)
                o_w[:, i] = -1 * output_w[:, i]
                o_b = torch.zeros_like(output_b)
                delta = torch.concatenate((i_w.flatten(), i_b.flatten(), o_w.flatten(), o_b.flatten()), dim=0)
                if self.nHiddenLayers == 1:
                    self.fisherUtility[i] = self.decayRate * self.fisherUtility[i] + (1 - self.decayRate) * torch.matmul(torch.matmul(torch.transpose(delta, 0, 1), self.F), delta)
                else:
                    self.fisherUtility[i, j] = self.decayRate * self.fisherUtility[i] + (1 - self.decayRate) * torch.matmul(torch.matmul(torch.transpose(delta, 0, 1), self.F), delta)

                if any(self.fisherUtility < 0):
                    print("ELEMENTS < 0 ------------------------------------------------------------------------------------------------------------")

    def FisherReset(self, mode='full'):

        weights = []
        biases = []
        for layer in self.layerList:
            if isinstance(layer, nn.Linear):
                layerWeights = layer.weight.detach()
                layerBias = layer.bias.detach()
                # Sum together contributions (sum elements of same columns) from same hidden unit
                # and reshape to obtain h1xh2 matrix to use in the formula.
                weights.append(layerWeights)
                biases.append(layerBias)

        # Do the same for each layer.
        for j in range(self.nHiddenLayers):
            # Select lower utility features depending on the replacement rate
            self.unitsToReplace[j] += self.replacementRate * np.count_nonzero(self.hiddenFisherUnitsAge > self.maturityThreshold)

            while (self.unitsToReplace[j] >= 1):
                # Scan matrix of utilities to find lower element with age > maturityThreshold.
                min = torch.amin(self.fisherUtility[:, j])
                minPos = []
                for i in range(self.fisherUtility.shape[0]):
                    if self.fisherUtility[i, j] == min and self.hiddenFisherUnitsAge[i, j] > self.maturityThreshold:
                        minPos.append(i)

                if not len(minPos):
                    break

                # Pick one position randomly
                minPos = np.random.choice(minPos)
                # Now out min and minPos values are legitimate, and we can replace the input weights and set
                # to zero the outgoing weights for the selected hidden unit.
                # Set to 0 the age of the hidden unit.
                self.hiddenFisherUnitsAge[minPos, j] = 0
                #self.hiddenUnitsCount[minPos, j] = 0
                # Set to 0 the utilities and mean values of the hidden unit.
                #self.hiddenUtilityBias[minPos, j] = 0
                #self.hiddenUnitsAvgBias[minPos, j] = 0
                self.fisherUtility[minPos, j] = 0

                # Reset weights
                # Input weights
                temp = nn.Linear(self.layerList[j].weight.shape[1], self.layerList[j].weight.shape[0])
                self.layerList[j].weight[minPos, :] = temp.weight.detach()[0, :]
                self.layerList[j].bias[minPos] = 0
                self.layerList[j + 1].weight[:, minPos] = 0

                # We replaced a hidden unit, reduce counter.
                self.unitsToReplace -= 1


    def growNet(self, no_of_neurons=1):

        if self.counter % self.growPeriod != 0 or self.counter == 0:
            self.counter += 1
            return
        with torch.no_grad():
            weights = []
            biases = []
            for layer in self.layerList:
                if isinstance(layer, nn.Linear):
                    layerWeights = layer.weight.detach()
                    layerBias = layer.bias.detach()
                    # Sum together contributions (sum elements of same columns) from same hidden unit
                    # and reshape to obtain h1xh2 matrix to use in the formula.
                    weights.append(layerWeights)
                    biases.append(layerBias)

            # Add the hidden unit
            self.l1 = torch.nn.Linear(self.I, self.H + no_of_neurons)
            self.l2 = torch.nn.Linear(self.H + no_of_neurons, self.outDim)

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
            self.model = nn.Sequential(self.l1, self.a1, self.l2)
            self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)

        # Add zero elements to Fisher matrix
        temp = np.zeros((self.F.shape[0] + no_of_neurons * (2 + self.I), self.F.shape[0]+ no_of_neurons * (2 + self.I))) # Add self.I params on l1, noofneurons params on l2 and b1
        tempIndex = 0
        fIndex = 0
        # Add entries relative to weights of first layer
        temp[tempIndex:tempIndex+weights[0].numel(), tempIndex:tempIndex+weights[0].numel()] = self.F[fIndex:fIndex+weights[0].numel(), fIndex:fIndex+weights[0].numel()]
        fIndex += weights[0].numel()
        tempIndex = fIndex + weights[0].shape[1] * no_of_neurons
        # Add entries relative to bias of first layer
        temp[tempIndex:tempIndex+biases[0].numel(), tempIndex:tempIndex+biases[0].numel()] = self.F[fIndex:fIndex+biases[0].numel(), fIndex:fIndex+biases[0].numel()]
        fIndex += biases[0].numel()
        tempIndex = fIndex + no_of_neurons
        # Add entries relative to weights of second layer
        temp[tempIndex:tempIndex+weights[1].numel(), tempIndex:tempIndex+weights[1].numel()] = self.F[fIndex:fIndex+weights[1].numel(), fIndex:fIndex+weights[1].numel()]
        fIndex += weights[1].numel()
        tempIndex = fIndex + weights[1].shape[0] * no_of_neurons
        # Add entries relative to bias of second layer
        temp[tempIndex:tempIndex+biases[1].numel(), tempIndex:tempIndex+biases[1].numel()] = self.F[fIndex:fIndex+biases[1].numel(), fIndex:fIndex+biases[1].numel()]

        self.F = temp
        # Add zero element to age vectors (both fisher and CBP)
        self.hiddenUnitsAge = np.append(self.hiddenUnitsAge, [0])
        self.hiddenFisherUnitsAge = np.append(self.hiddenFisherUnitsAge, [0])
        # Add zero elements to utility vectors of CBP and Fisher
        self.hiddenUnits = np.append(self.hiddenUnits, [0])
        self.hiddenUnitsAvg = np.append(self.hiddenUnitsAvg, [0])
        self.hiddenUnitsAvgBias = np.append(self.hiddenUnitsAvgBias, [0])
        self.hiddenUtilityBias = np.append(self.hiddenUtilityBias, [0])
        self.hiddenUtility = np.append(self.hiddenUtility, [0])
        self.fisherUtility = np.append(self.fisherUtility, [0])

        self.counter += 1
        return

if __name__ == "__main__":
    model = Actor(3,4)
    input = torch.tensor([1., 2., 3.])

    #print(model.parameters())

    for param in model.parameters():
        #print("Start printing parameters:")
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
        #print("Start printing parameters:")
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

    #print(model.state_dict()['l2.weight '])


