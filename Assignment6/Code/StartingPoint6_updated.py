## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import numpy as np
## NOTE update this with your equivalent code..
import TrainTestSplit

from PIL import Image
import numpy as np

#predictionThreshold = .5

thresholds = [.5]
for i in range(0,101):
    thresholds.append(i*.01)

kDataPath = "..\\Datasets\\FaceData\\dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))


print("Calculating features...")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeXGradients=False, includeYGradients=False, includeYHistogram=False, includeXHistogram=False, includeRawPixels=False, includeIntensities=True)
yTrain = yTrainRaw
yTest = yTestRaw

np_xTrain = np.array(xTrain)
np_yTrain = np.array(yTrain)
np_xTest = np.array(xTest)
np_yTest = np.array(yTest)

import Evaluations
import ErrorBounds

######
import MostCommonModel
model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)
print("Most Common Accuracy:", Evaluations.Accuracy(yTest, yTestPredicted), ErrorBounds.Get95LowerAndUpperBounds(Evaluations.Accuracy(yTest, yTestPredicted), len(yTest)))

######
import NeuralNetworkModel

hiddenLayers = [2]
hiddenNodes = [2, 5, 10, 15, 20]

#hiddenLayers = 2
#hiddenNodes = 3
#sample = [1, .75, 1.5, 1]

#neuralNetwork = NeuralNetworkModel.NeuralNetworkModel(hiddenLayers, hiddenNodes, len(xTrain[0]))
#neuralNetwork.UnitTest()
#neuralNetwork.fit(xTrain, yTrain, iterations=200, step=0.05))
for layerCount in hiddenLayers:
    for nodeCount in hiddenNodes:
        print("Loss calculated for layer count:", layerCount," node count: ",nodeCount)
        neuralNetwork = NeuralNetworkModel.NeuralNetworkModel(layerCount, nodeCount, len(xTrain[0]))
        #neuralNetwork.fit(xTrain, yTrain, xTest, yTest, iterations=200, step=0.05)
neuralNetwork.UnitTestTwo()