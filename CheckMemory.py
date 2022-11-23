import torch
import numpy as np
from Algorithms.LearningNet import LearningNet
from Algorithms.GrowingNet import GrowingNet
from Algorithms.FisherNet import FisherNet
from Algorithms.FisherUnitNet import FisherUnitNet
from Algorithms.TargetNet import TargetNet
from Algorithms.RandomResNet import RandomResNet
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
randErrors = np.zeros((runsN, exampleN))
fisherErrors = np.zeros((runsN, exampleN))
fisherUnitErrors = np.zeros((runsN, exampleN))
benchErrors = np.zeros((runsN, exampleN))
inputs = np.zeros((runsN, int(exampleN/2), m))
outputs = np.zeros((runsN, int(exampleN/2)))

# Target network
target = TargetNet(m)

# Loop of doing regression, evaluating error, training learners
for j in range(0, runsN):

    # Create learner networks
    contLearner = LearningNet(m, 1)
    randLearner = RandomResNet(m, 1)
    benchLearner = LearningNet(m, 1)
    fisherLearner = FisherNet(m, 1)
    fisherUnitLearner = FisherUnitNet(m, 1)

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
        y3 = y.detach().clone()
        y4 = y.detach().clone()
        y5 = y.detach().clone()
        outputs[j, i] = y.detach().item()

        outCont = contLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outBench = benchLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outRand = randLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outFisher = fisherLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outFisherUnit = fisherUnitLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))

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

        # Train random learner
        randLoss = (outRand - y3) ** 2
        randErrors[j, i] = randLoss.detach().item()
        randLearner.zero_grad()
        randLoss.backward()
        randLearner.optimizer.step()
        randLearner.genAndTest()

        # Train Fisher learner
        fisherLoss = (outFisher - y4) ** 2
        fisherErrors[j, i] = fisherLoss.detach().item()
        fisherLearner.zero_grad()
        fisherLoss.backward()
        fisherLearner.optimizer.step()
        fisherLearner.computeFisher()
        fisherLearner.FisherReplacement()

        # Train FisherUnit learner
        fisherUnitLoss = (outFisherUnit - y5) ** 2
        fisherUnitErrors[j, i] = fisherUnitLoss.detach().item()
        fisherUnitLearner.zero_grad()
        fisherUnitLoss.backward()
        fisherUnitLearner.optimizer.step()
        fisherUnitLearner.genAndTest()

    # Feed old values
    for i in range(int(exampleN/2), exampleN):
        outCont = contLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outBench = benchLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outRand = randLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))
        outFisher = fisherLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))
        outFisherUnit = fisherUnitLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))

        y = target(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        y1 = y.detach().clone()
        y2 = y.detach().clone()
        y3 = y.detach().clone()
        y4 = y.detach().clone()
        y5 = y.detach().clone()

        contLoss = (outCont - y1) ** 2
        benchLoss = (outBench - y2) ** 2
        randLoss = (outRand - y3) ** 2
        fisherLoss = (outFisher - y4) ** 2
        fisherUnitLoss = (outFisherUnit - y5) ** 2

        contErrors[j, i] = contLoss.detach().item()
        randErrors[j, i] = randLoss.detach().item()
        benchErrors[j, i] = benchLoss.detach().item()
        fisherErrors[j, i] = fisherLoss.detach().item()
        fisherUnitErrors[j, i] = fisherUnitLoss.detach().item()

        '''# Train cont
        contLearner.zero_grad()
        contLoss.backward()
        contLearner.optimizer.step()
        contLearner.genAndTest()
        # Train bench
        benchLearner.zero_grad()
        benchLoss.backward()
        benchLearner.optimizer.step()
        # Train grow
        growLearner.zero_grad()
        growLoss.backward()
        growLearner.optimizer.step()
        growLearner.growNet(1)
        
        # Train fisher
        fisherLearner.zero_grad()
        fisherLoss.backward()
        fisherLearner.computeFisher()
        fisherLearner.optimizer.step()
        fisherLearner.FisherReplacement()'''

np.save("contErrors", contErrors)
np.save("randErrors", randErrors)
np.save("benchErrors", benchErrors)
np.save("fisherErrors", fisherErrors)
np.save("fisherUnitErrors", fisherUnitErrors)
np.save("inputHistory", inputs)
np.save("outputHistory", outputs)





