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

exampleN = 1000000
runsN = 5

# Errors values
contErrors = np.zeros((runsN, exampleN))
benchErrors = np.zeros((runsN, exampleN))
inputs = np.zeros((runsN, exampleN, m))


# Loop of doing regression, evaluating error, training learners
for j in range(0, runsN):

    # Create learner networks and target network
    target = TargetNet(m)
    contLearner = LearningNet(m, 1)
    benchLearner = LearningNet(m, 1)
    # Set input
    input = np.random.choice([0, 1], m)

    for i in range(0, exampleN):
        if i % T == 0:
            print("Run {}, iteration {}".format(j, i))
        # Sample the input
        # Set the random
        input[f:] = np.random.choice([0,1], m-f)
        # Flip one of the first f bit every T timesteps
        if i % T == 0:
            index = np.random.randint(f)
            if input[index] == 0:
                input[index] = 1
            elif input[index] == 1:
                input[index] = 0
            else:
                print("This is very wrong")
                sys.exit()

        # Save input
        inputs[j,i,:] = input
        # Now the input is ready to be used
        # Get true value
        y = target(torch.from_numpy(input).type(torch.FloatTensor))
        y1 = y.detach().clone()
        y2= y.detach().clone()

        outCont = contLearner(torch.from_numpy(input).type(torch.FloatTensor))
        outBench = benchLearner(torch.from_numpy(input).type(torch.FloatTensor))

        # Train contLearner
        contLoss = (outCont - y1) ** 2
        contErrors[j, i] = contLoss
        contLearner.zero_grad()
        contLoss.backward()
        contLearner.optimizer.step()
        contLearner.genAndTest()

        # Train benchLearner
        benchLoss = (outBench - y2) ** 2
        benchErrors[j, i] = benchLoss
        benchLearner.zero_grad()
        benchLoss.backward()
        benchLearner.optimizer.step()

np.save("contErrors", contErrors)
np.save("benchErrors", benchErrors)
np.save("inputHistory", inputs)




