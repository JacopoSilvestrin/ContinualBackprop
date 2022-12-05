import torch
import numpy as np
from Algorithms.LearningNet import LearningNet
from Algorithms.GrowingNet import GrowingNet
from Algorithms.FisherNet import FisherNet
from Algorithms.FisherUnitNet import FisherUnitNet
from Algorithms.TargetNet import TargetNet
from Algorithms.RandomResNet import RandomResNet
from Algorithms.DetectingNet import DetectingNet
from Algorithms.GrowCBP import GrowCBP
import random
import sys
from torch import nn

# Bit-Flipping problem
m = 20
f = 15
T = 10000

# Detection parameters

# Set seed
random.seed(42)
np.random.seed(42)

exampleN = 60000
runsN = 30

# Errors values
contErrors = np.zeros((runsN, exampleN))
fisherUnitErrors = np.zeros((runsN, exampleN))
benchErrors = np.zeros((runsN, exampleN))
growContErrors = np.zeros((runsN, exampleN))
bigContErrors = np.zeros((runsN, exampleN))
detErrors = np.zeros((runsN, exampleN))
inputs = np.zeros((runsN, int(exampleN/2), m))
outputs = np.zeros((runsN, int(exampleN/2)))

# Target network
target = TargetNet(m)

# Loop of doing regression, evaluating error, training learners
for j in range(0, runsN):

    # Create learner networks
    benchLearner = LearningNet(m, 1)
    contLearner = LearningNet(m, 1)
    growContLearner = GrowCBP(m, 1)
    bigContLearner = LearningNet(m, 1, hidden_dim=10)
    fisherUnitLearner = FisherUnitNet(m, 1)
    detLearner = DetectingNet(m, 1)
    #randLearner = RandomResNet(m, 1)
    #fisherLearner = FisherNet(m, 1)

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
        y6 = y.detach().clone()
        outputs[j, i] = y.detach().item()

        outBench = benchLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outCont = contLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outGrowCont = growContLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outBigCont = bigContLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outFisherUnit = fisherUnitLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outDet = detLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        #outRand = randLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        #outFisher = fisherLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))

        # Train benchLearner
        benchLoss = (outBench - y1) ** 2
        benchErrors[j, i] = benchLoss.detach().item()
        benchLearner.zero_grad()
        benchLoss.backward()
        benchLearner.optimizer.step()

        # Train contLearner
        contLoss = (outCont - y2) ** 2
        contErrors[j, i] = contLoss.detach().item()
        contLearner.zero_grad()
        contLoss.backward()
        contLearner.optimizer.step()
        contLearner.genAndTest()

        # Train GrowCont
        growContLoss = (outGrowCont - y3) ** 2
        growContErrors[j, i] = growContLoss.detach().item()
        growContLearner.zero_grad()
        growContLoss.backward()
        growContLearner.optimizer.step()
        growContLearner.genAndTest()
        growContLearner.growNet()

        # Train BigCont
        bigContLoss = (outBigCont - y4) ** 2
        bigContErrors[j, i] = bigContLoss.detach().item()
        bigContLearner.zero_grad()
        bigContLoss.backward()
        bigContLearner.optimizer.step()
        bigContLearner.genAndTest()

        # Train random learner
        #randLoss = (outRand - y3) ** 2
        #randErrors[j, i] = randLoss.detach().item()
        #randLearner.zero_grad()
        #randLoss.backward()
        #randLearner.optimizer.step()
        #randLearner.genAndTest()

        # Train Fisher learner
        #fisherLoss = (outFisher - y4) ** 2
        #fisherErrors[j, i] = fisherLoss.detach().item()
        #fisherLearner.zero_grad()
        #fisherLoss.backward()
        #fisherLearner.optimizer.step()
        #fisherLearner.computeFisher()
        #fisherLearner.FisherReplacement()

        # Train FisherUnit learner
        fisherUnitLoss = (outFisherUnit - y5) ** 2
        fisherUnitErrors[j, i] = fisherUnitLoss.detach().item()
        fisherUnitLearner.zero_grad()
        fisherUnitLoss.backward()
        fisherUnitLearner.optimizer.step()
        fisherUnitLearner.genAndTest()

        # Train DetNetwork learner
        detLoss = (outDet - y6) ** 2
        detErrors[j, i] = detLoss.detach().item()
        detLearner.zero_grad()
        detLoss.backward()
        detLearner.optimizer.step()
        detLearner.genAndTest()
        detLearner.growNet()

    # Feed old values
    for i in range(int(exampleN/2), exampleN):
        outCont = contLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outGrowCont = growContLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outBigCont = bigContLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outBench = benchLearner(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        outFisherUnit = fisherUnitLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))
        outDet = detLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))
        #outRand = randLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))
        #outFisher = fisherLearner(torch.from_numpy(inputs[j, i - int(exampleN / 2)]).type(torch.FloatTensor))

        y = target(torch.from_numpy(inputs[j,i-int(exampleN/2)]).type(torch.FloatTensor))
        y1 = y.detach().clone()
        y2 = y.detach().clone()
        y3 = y.detach().clone()
        y4 = y.detach().clone()
        y5 = y.detach().clone()
        y6 = y.detach().clone()

        benchLoss = (outBench - y1) ** 2
        contLoss = (outCont - y2) ** 2
        growContLoss = (outGrowCont - y3) ** 2
        bigContLoss = (outBigCont - y4) ** 2
        fisherUnitLoss = (outFisherUnit - y5) ** 2
        detLoss = (outDet - y6) ** 2
        #randLoss = (outRand - y3) ** 2
        #fisherLoss = (outFisher - y4) ** 2

        benchErrors[j, i] = benchLoss.detach().item()
        contErrors[j, i] = contLoss.detach().item()
        growContErrors[j, i] = growContLoss.detach().item()
        bigContErrors[j, i] = bigContLoss.detach().item()
        fisherUnitErrors[j, i] = fisherUnitLoss.detach().item()
        detErrors[j, i] = detLoss.detach().item()
        #fisherErrors[j, i] = fisherLoss.detach().item()
        #randErrors[j, i] = randLoss.detach().item()

        # Train bench
        benchLearner.zero_grad()
        benchLoss.backward()
        benchLearner.optimizer.step()

        # Train cont
        contLearner.zero_grad()
        contLoss.backward()
        contLearner.optimizer.step()
        contLearner.genAndTest()

        # Train GrowCont
        growContLearner.zero_grad()
        growContLoss.backward()
        growContLearner.optimizer.step()
        growContLearner.genAndTest()
        growContLearner.growNet()

        # Train BigCont
        bigContLearner.zero_grad()
        bigContLoss.backward()
        bigContLearner.optimizer.step()
        bigContLearner.genAndTest()

        # Train DetNetwork learner
        detLearner.zero_grad()
        detLoss.backward()
        detLearner.optimizer.step()
        detLearner.genAndTest()
        detLearner.growNet()

        # Train unit fisher
        fisherUnitLearner.zero_grad()
        fisherUnitLoss.backward()
        fisherUnitLearner.optimizer.step()
        fisherUnitLearner.genAndTest()

        # Train grow
        #randLearner.zero_grad()
        #randLoss.backward()
        #randLearner.optimizer.step()
        #randLearner.genAndTest()
        # Train fisher
        #fisherLearner.zero_grad()
        #fisherLoss.backward()
        #fisherLearner.computeFisher()
        #fisherLearner.optimizer.step()
        #fisherLearner.FisherReplacement()

np.save("contErrors", contErrors)
np.save("growContErrors", growContErrors)
np.save("bigContErrors", bigContErrors)
np.save("benchErrors", benchErrors)
np.save("fisherUnitErrors", fisherUnitErrors)
np.save("detErrors", detErrors)
np.save("inputHistory", inputs)
np.save("outputHistory", outputs)
#np.save("randErrors", randErrors)
#np.save("fisherErrors", fisherErrors)





