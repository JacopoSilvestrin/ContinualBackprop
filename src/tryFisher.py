import torch
import numpy as np
from Algorithms.LearningNet import LearningNet
from Algorithms.FisherNet import FisherNet
from Algorithms.TargetNet import TargetNet
from Algorithms.GrowingNet import GrowingNet
import random
import sys
from torch.utils.data import TensorDataset, DataLoader
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag
from torch import nn

m = 20
f = 15
T = 10000
# Set seed
random.seed(42)
np.random.seed(42)

exampleN = 10000
runsN = 1

# Target network
target = TargetNet(m)
inputs = np.zeros((runsN, exampleN, m))
l1w_grad_mean = np.zeros((5, m))
l2w_grad_mean = np.zeros((1, 5))
l1b_grad_mean = np.zeros(5)
l2b_grad_mean = 0

for j in range(0, runsN):

    # Create learner networks
    fisherLearner = FisherNet(m, 1)

    # Set input
    inputVec = np.random.choice([0, 1], m)

    for i in range(0, exampleN):

        if i % T == 0:
            print("Run {}, iteration {}".format(j, i))
        # Sample the input
        # Set the random
        inputVec[f:] = np.random.choice([0, 1], m - f)
        # Flip one of the first f bit every T timesteps
        if i % T == 0 and i != 0:
            index = np.random.randint(f)
            if inputVec[index] == 0:
                inputVec[index] = 1
            elif inputVec[index] == 1:
                inputVec[index] = 0
            else:
                print("This is very wrong")
                sys.exit()

        # Save input
        inputs[j, i, :] = inputVec
        # Now the input is ready to be used
        # Get true value
        y = target(torch.from_numpy(inputVec).type(torch.FloatTensor))
        y1 = y.detach().clone()

        outCont = fisherLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))


        # Train contLearner
        contLoss = (outCont - y1) ** 2
        fisherLearner.zero_grad()
        contLoss.backward()

        # Computation of weight importance
        fisherLearner.computeFisher()
        '''paramsVec = np.array([])
        for layer in contLearner.model:
            if isinstance(layer, nn.Linear):
                weightsGrad = layer.weight.grad.detach().flatten().numpy()
                biasGrad = layer.bias.grad.detach().flatten().numpy()
                paramsVec = np.concatenate((paramsVec, weightsGrad, biasGrad), axis=0)

        contLearner.F += np.outer(paramsVec, paramsVec)
        contLearner.DiagF += np.square(paramsVec)
        contLearner.FCount += 1'''


        # Save gradient values
        l1w_grad_mean += fisherLearner.l1.weight.grad.detach().numpy()
        l1b_grad_mean += fisherLearner.l1.bias.grad.detach().numpy()
        l2w_grad_mean += fisherLearner.l2.weight.grad.detach().numpy()
        l2b_grad_mean += fisherLearner.l2.bias.grad.detach().numpy()

        fisherLearner.optimizer.step()
        fisherLearner.FisherReplacement()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(inputs[j,:,:]).type(torch.FloatTensor))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    #FIMLib = FIM(model=contLearner, loader=loader, representation=PMatDiag, n_output=1, variant='regression')
    #FIMmia = fisherLearner.F / fisherLearner.FCount
    #FIMDiag = fisherLearner.DiagF / fisherLearner.FCount
    l1w_grad_mean = l1w_grad_mean / exampleN
    l1b_grad_mean = l1b_grad_mean / exampleN
    l2w_grad_mean = l2w_grad_mean / exampleN
    l2b_grad_mean = l2b_grad_mean / exampleN



    print("FIM:")
    #print(F.data)

    print("Print Params")
    print(fisherLearner.l1.weight)
    print(fisherLearner.l1.bias)
    print(fisherLearner.l2.weight)
    print(fisherLearner.l2.bias)

    print("Print grads mean")
    print(l1w_grad_mean)
    print(l1b_grad_mean)
    print(l2w_grad_mean)
    print(l2b_grad_mean)


