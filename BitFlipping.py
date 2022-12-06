import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import torch
import numpy as np
from Algorithms.LearningNet import LearningNet
from Algorithms.TargetNet import TargetNet
from Algorithms.GrowingNet import GrowingNet
from Algorithms.FisherNet import FisherNet
from Algorithms.GrowCBP import GrowCBP
from Algorithms.FisherUnitNet import FisherUnitNet
from Algorithms.DetectingNet import DetectingNet
import random
import sys
from torch import nn


torch.set_num_threads(1)


# Bit-Flipping problem

m = 20
f = 15
T = 10000

# Set seed
random.seed(42)
np.random.seed(42)

exampleN = 1000000
runsN = 30

# Errors values
contErrors = np.zeros((runsN, exampleN))
fisherUnitErrors = np.zeros((runsN, exampleN))
benchErrors = np.zeros((runsN, exampleN))
growContErrors = np.zeros((runsN, exampleN))
bigContErrors = np.zeros((runsN, exampleN))
growFisherErrors = np.zeros((runsN, exampleN))
inputs = np.zeros((runsN, exampleN, m))
outputs = np.zeros((runsN, exampleN))

# Target network
target = TargetNet(m)


# Loop of doing regression, evaluating error, training learners
for j in range(0, runsN):

    # Create learner networks
    benchLearner = LearningNet(m, 1)
    contLearner = LearningNet(m, 1)
    growContLearner = GrowCBP(m, 1)
    bigContLearner = LearningNet(m, 1, hidden_dim=105)
    fisherUnitLearner = FisherUnitNet(m, 1)
    detLearner = DetectingNet(m, 1)

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
            print("Let's flip a bit.")
            index = np.random.randint(f)
            if inputVec[index] == 0:
                inputVec[index] = 1
            elif inputVec[index] == 1:
                inputVec[index] = 0
            else:
                print("This is very wrong")
                sys.exit()
        
        
        # Save input
        inputs[j,i,:] = inputVec
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

        # Train FisherUnit learner
        fisherUnitLoss = (outFisherUnit - y5) ** 2
        fisherUnitErrors[j, i] = fisherUnitLoss.detach().item()
        fisherUnitLearner.zero_grad()
        fisherUnitLoss.backward()
        fisherUnitLearner.optimizer.step()
        fisherUnitLearner.genAndTest()

        # Train DetNetwork learner
        detLoss = (outDet - y6) ** 2
        growFisherErrors[j, i] = detLoss.detach().item()
        detLearner.zero_grad()
        detLoss.backward()
        detLearner.optimizer.step()
        detLearner.genAndTest()
        detLearner.growNet()

np.save("contErrors", contErrors)
np.save("growContErrors", growContErrors)
np.save("bigContErrors", bigContErrors)
np.save("benchErrors", benchErrors)
np.save("fisherUnitErrors", fisherUnitErrors)
np.save("detErrors", growFisherErrors)
np.save("inputHistory", inputs)
np.save("outputHistory", outputs)





