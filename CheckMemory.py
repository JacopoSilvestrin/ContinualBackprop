import torch
import numpy as np
from Algorithms.LearningNet import LearningNet
from Algorithms.TargetNet import TargetNet
import random
import sys
from torch import nn

# Bit-Flipping problem

m = 20
f = 15
T = 10000

# Set seed
random.seed(42)
np.random.seed(42)

exampleN = 60000
runsN = 30

# Errors values
contErrors = np.zeros((runsN, exampleN))
benchErrors = np.zeros((runsN, exampleN))
inputs = np.zeros((runsN, int(exampleN/2), m))
outputs = np.zeros((runsN, int(exampleN/2)))

# Target network
target = TargetNet(m)

# Loop of doing regression, evaluating error, training learners
for j in range(0, runsN):

    # Create learner networks
    contLearner = LearningNet(m, 1)
    benchLearner = LearningNet(m, 1)

    # Set input
    inputVec = np.random.choice([0, 1], m)

    for i in range(0, int(exampleN/2)):
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
        y2 = y.detach().clone()
        outputs[j, i] = y.detach().item()

        outCont = contLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outBench = benchLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))

        # Train contLearner
        contLoss = (outCont - y1) ** 2
        contErrors[j, i] = contLoss.detach().item()
        contLearner.zero_grad()
        contLoss.backward()
        contLearner.optimizer.step()
        contLearner.genAndTest()

        # Train benchLearner
        benchLoss = (outBench - y2) ** 2
        benchErrors[j, i] = benchLoss.detach().item()
        benchLearner.zero_grad()
        benchLoss.backward()
        benchLearner.optimizer.step()

    # Feed old values
    for i in range(int(exampleN/2), exampleN):
        outCont = contLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outBench = benchLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        y = target(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        y1 = y.detach().clone()
        y2 = y.detach().clone()
        contLoss = (outCont - y1) ** 2
        benchLoss = (outBench - y2) ** 2
        contErrors[j, i] = contLoss.detach().item()
        benchErrors[j, i] = benchLoss.detach().item()

        # Train cont
        contLearner.zero_grad()
        contLoss.backward()
        contLearner.optimizer.step()
        contLearner.genAndTest()
        # Train bench
        benchLearner.zero_grad()
        benchLoss.backward()
        benchLearner.optimizer.step()


np.save("contErrors", contErrors)
np.save("benchErrors", benchErrors)
np.save("inputHistory", inputs)
np.save("outputHistory", outputs)





