## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')
## Remember to install Pytorch: https://pytorch.org/get-started/locally/ (if you want GPU you need to figure out CUDA...)


import Assignment5Support

## NOTE update this with your equivalent code.
import TrainTestSplit

kDataPath = "..\\Datasets\\FaceData\\dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

from PIL import Image
import torch 
import torchvision.transforms as transforms
import CrossValidationSupport

torch.cuda.set_device(0)

print(torch.cuda.is_available())
torch.cuda.manual_seed(0)

#torch.cuda.device(torch.device('cuda'))

xTrainRaw, yTrainRaw, xValidateRaw, yValidateRaw = CrossValidationSupport.DefineDataBounds(xTrainRaw, yTrainRaw, 10, 0)

xValidateImages = [ Image.open(path).resize((112,112)) for path in xValidateRaw ]
xValidate = torch.stack([ transforms.ToTensor()(image) for image in xValidateImages ]).cuda()

yValidate = torch.Tensor([ [ yValue ] for yValue in yValidateRaw ]).cuda()

# Load the images and then convert them into tensors (no normalization)
xTrainImages = [ Image.open(path).resize((112,112)) for path in xTrainRaw ]
xTrain = torch.stack([ transforms.ToTensor()(image) for image in xTrainImages ]).cuda()

xTestImages = [ Image.open(path).resize((112,112)) for path in xTestRaw ]
xTest = torch.stack([ transforms.ToTensor()(image) for image in xTestImages ]).cuda()

yTrain = torch.Tensor([ [ yValue ] for yValue in yTrainRaw ]).cuda()
yTest = torch.Tensor([ [ yValue ] for yValue in yTestRaw ]).cuda()

import Evaluations
import ErrorBounds
"""
######
######


import SimpleBlinkNeuralNetwork
import UpdatedBlinkNeuralNetwork

# Create the model and set up:
#     the loss function to use (Mean Square Error)
#     the optimization method (Stochastic Gradient Descent) and the step size
model = UpdatedBlinkNeuralNetwork.ThreeLayerBlinkNeuralNetwork(hiddenNodes = 5)
lossFunction = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

lookaheadCount = 2
currAcc = 0
bestAcc = -1
i = 0
currLookahead = 0

########################################

while currLookahead < lookaheadCount:
    # Do the forward pass
    yTrainPredicted = model(xTrain).cuda()

    # Compute the training set loss
    loss = lossFunction(yTrainPredicted, yTrain).cuda()
    print(i, loss.item())
    
    # Reset the gradients in the network to zero
    optimizer.zero_grad()

    # Backprop the errors from the loss on this iteration
    loss.backward()

    # Do a weight update step
    optimizer.step()

    if i%10 == 0:
        currLookahead += 1
        xValidatePredicted = model(xValidate).cuda()

        xPred = [ 1 if pred > 0.5 else 0 for pred in xValidatePredicted ]

        #print("Acccuracy:", Evaluations.Accuracy(yValidate, xPred))
        currAcc = Evaluations.Accuracy(yValidate, xPred)

        if currAcc >= bestAcc:
            bestAcc = currAcc
            currLookahead = 0
            bestModel = model

    i += 1
yTestPredicted = bestModel(xTest).cuda()

yPred = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]

currTotalAcc = Evaluations.Accuracy(yTest, yPred)

print("Best accuracy Alex now at :", currTotalAcc)

#######################


for i in range(500):
    # Do the forward pass
    yTrainPredicted = model(xTrain).cuda()

    # Compute the training set loss
    loss = lossFunction(yTrainPredicted, yTrain).cuda()
    print(i, loss.item())
    
    # Reset the gradients in the network to zero
    optimizer.zero_grad()

    # Backprop the errors from the loss on this iteration
    loss.backward()

    # Do a weight update step
    optimizer.step()

yTestPredicted = model(xTest).cuda()

yPred = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]

print("Accuracy Alex:", Evaluations.Accuracy(yTest, yPred))
currTotalAcc = Evaluations.Accuracy(yTest, yPred)
"""

y_len = len(yTest)
accuracy = 0.9661716171617162
(lowerBounds, upperBounds) = Evaluations.calculate95PercentConfidenceBounds(accuracy, y_len)
print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)