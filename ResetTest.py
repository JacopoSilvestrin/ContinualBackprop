import torch
import numpy as np
from Algorithms.LearningNet import LearningNet
from Algorithms.TargetNet import TargetNet
from Algorithms.FisherUnitNet import FisherUnitNet
import random
import sys
from torch import nn


def pickNewPos(utility, picked):
    min = np.amax(utility)
    minPos = np.argmax(utility)
    for i in range(utility.size):
        if utility[i] <= min and i not in picked:
            min = utility[i]
            minPos = i

    if minPos not in picked:
        picked.append(minPos)
        return minPos, picked
    else:
        # nothing to reset
        return None, picked


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
fisherUnitErrors = np.zeros((runsN, exampleN))
inputs = np.zeros((runsN, exampleN, m))
outputs = np.zeros((runsN, exampleN))

# Target network
target = TargetNet(m)

# Loop of doing regression, evaluating error, training learners
for j in range(0, runsN):

    # Create learner networks
    contLearner = LearningNet(m, 1)
    fisherUnitLearner = FisherUnitNet(m, 1)
    contUtility = None
    fisherUtility = None
    contPos = [0, 1, 2, 3, 4]
    fisherPos = [0, 1, 2, 3, 4]
    contPicked = []
    fisherPicked = []

    # Storage for network original params
    contL1 = None
    contB1 = None
    contL2 = None
    fisherL1 = None
    fisherB1 = None
    fisherL2 = None



    # Set input
    inputVec = np.random.choice([0, 1], m)

    for i in range(0, exampleN):
        if i % T == 0:
            print("Run {}, iteration {}".format(j, i))
        # Sample the input
        # Set the random
        inputVec[f:] = np.random.choice([0, 1], m - f)

        # Save input
        inputs[j, i, :] = inputVec
        # Now the input is ready to be used
        # Get true value
        y = target(torch.from_numpy(inputVec).type(torch.FloatTensor))
        y1 = y.detach().clone()
        y2 = y.detach().clone()

        outputs[j, i] = y.detach().item()

        outCont = contLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        outFisher = fisherUnitLearner(torch.from_numpy(inputVec).type(torch.FloatTensor))
        # Train normally first 10k samples
        if i < 10000:
            # Train contLearner
            contLoss = (outCont - y1) ** 2
            contErrors[j, i] = contLoss.detach().item()
            contLearner.zero_grad()
            contLoss.backward()
            contLearner.optimizer.step()
            contLearner.genAndTest()
            contUtility = contLearner.hiddenUtility

            # Train Fisher learner
            fisherLoss = (outFisher - y2) ** 2
            fisherUnitErrors[j, i] = fisherLoss.detach().item()
            fisherUnitLearner.zero_grad()
            fisherLoss.backward()
            fisherUnitLearner.genAndTest()
            fisherUtility = fisherUnitLearner.fisherUtility

            # Update weight and bias stored values
            contL1 = contLearner.l1.weight.data.detach()
            contB1 = contLearner.l1.bias.data.detach()
            contL2 = contLearner.l2.weight.data.detach()
            fisherL1 = fisherUnitLearner.l1.weight.data.detach()
            fisherB1 = fisherUnitLearner.l1.bias.data.detach()
            fisherL2 = fisherUnitLearner.l2.weight.data.detach()
        else:
            # Pick next position to replace
            if i % T == 0:
                contMinPos, contPicked = pickNewPos(contUtility, contPicked)
                fisherMinPos, fisherPicked = pickNewPos(fisherUtility, fisherPicked)
                print("At iteration {} picked unit {} for CBP and {} for FBP".format(i, contMinPos, fisherMinPos))
                if contMinPos is not None and fisherMinPos is not None:
                    print("I am setting the weights to zero.")

                    # Reset cont weights
                    # Reset weights
                    # Take state_dict
                    weights = contLearner.state_dict()
                    # Reinitialise input weights (i-th row of previous layer)
                    weights['l1.weight'] = torch.clone(contL1)
                    weights['l1.weight'][contMinPos, :] = 0
                    # Reset the input bias
                    weights['l1.bias'] = torch.clone(contB1)
                    weights['l1.bias'][contMinPos] = 0
                    # Set to 0 outgoing weights (i-th column of next layer) and do the same for bias
                    weights['l2.weight'] = torch.clone(contL2)
                    weights['l2.weight'][:, contMinPos] = 0
                    # weights['l2.bias'][i] = 0
                    # Load stat_dict to the model to save changes
                    contLearner.load_state_dict(weights)

                    # Reset Fisher weighs
                    # Reset weights
                    # Take state_dict
                    weights = fisherUnitLearner.state_dict()
                    # Reinitialise input weights (i-th row of previous layer)
                    weights['l1.weight'] = torch.clone(fisherL1)
                    weights['l1.weight'][fisherMinPos, :] = 0
                    # Reset the input bias
                    weights['l1.bias'] = torch.clone(fisherB1)
                    weights['l1.bias'][fisherMinPos] = 0
                    # Set to 0 outgoing weights (i-th column of next layer) and do the same for bias
                    weights['l2.weight'] = torch.clone(fisherL2)
                    weights['l2.weight'][:, fisherMinPos] = 0
                    # weights['l2.bias'][i] = 0
                    # Load stat_dict to the model to save changes
                    fisherUnitLearner.load_state_dict(weights)
                else:
                    print("I already resetted all the weights:")
                    print(contPicked)
                    print(fisherPicked)

                # Evaluate without training
                # Train contLearner
                contLoss = (outCont - y1) ** 2
                contErrors[j, i] = contLoss.detach().item()

                # Train Fisher learner
                fisherLoss = (outFisher - y2) ** 2
                fisherUnitErrors[j, i] = fisherLoss.detach().item()

np.save("contErrors", contErrors)
np.save("fisherUnitErrors", fisherUnitErrors)
np.save("inputHistory", inputs)
np.save("outputHistory", outputs)







