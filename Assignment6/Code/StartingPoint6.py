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
import copy
import CrossValidationSupport
import Evaluations
import ErrorBounds
import NeuralNetworkModel

predictionThreshold = .5

#thresholds = [.5]
#for i in range(0,101):
#    thresholds.append(i*.01)

kDataPath = "..\\Datasets\\FaceData\\dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))


print("Calculating features...")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeXGradients=False, includeYGradients=False, includeYHistogram=False, includeXHistogram=False, includeRawPixels=False, includeIntensities=True)
yTrain = yTrainRaw
yTest = yTestRaw

k = 5
UseCrossValidation = True

for i in range(k):

    #print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
    #print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))
    if UseCrossValidation == True:
        print("Cross validation on ",i)
        (xTrain, yTrain, xValidate, yValidate) = CrossValidationSupport.DefineDataBounds(xTrain, yTrain, k, i)
        #(xTrain, xValidate) = Assignment1Support.Featurize(xTrainOnRaw, yTrainOnRaw, xValidateOnRaw, bagModel, numFrequentWords=frequentWordCount, numMutualInformationWords=mutualInfo, includeHandCraftedFeatures=UseHandcraftedFeatures)

        np_xTrain = np.asarray(xTrain)
        np_xTest = np.asarray(xValidate)

        np_yTrain = np.asarray(yTrain)
        np_yTest = np.asarray(yValidate)
    else:  
        np_xTrain = np.asarray(xTrain)
        np_xTest = np.asarray(xTest)
        np_yTrain = np.asarray(yTrain)
        np_yTest = np.asarray(yTest)
        
    hiddenLayers = [1]
    hiddenNodes = [2, 5, 10, 15, 20, 25]
    stepSizes = [25, 150]
    #stepSizes = [200]
    #hiddenLayers = [1]
    #hiddenNodes = [10]

    for layerCount in hiddenLayers:
        for nodeCount in hiddenNodes:
            for currStep in stepSizes:
                    textFile = "node" + str(nodeCount) + "step" + str(currStep) + "_two.txt"
                    currStep *= .001
                    with open(textFile, 'a') as currFile:
                        currFile.write("Loss calculated for layer count:" + str(layerCount) + " node count: " + str(nodeCount) +" stepSize:" + str(currStep)+" Momentum: False\n")
                        neuralNetwork = NeuralNetworkModel.NeuralNetworkModel(layerCount, nodeCount, len(xTrain[0]), addMomentum=False)
                        initialWeights = copy.deepcopy(neuralNetwork.nodeWeight)
                        yTestPredicted = neuralNetwork.fit(xTrain, yTrain, xTest, yTest, iterations=501, step=currStep, results=True, outputFile=currFile)
                        accuracy = Evaluations.Accuracy(yTest, yTestPredicted)
                        y_len =  len(yTestPredicted)
                        (lowerBounds, upperBounds) = ErrorBounds.Get95LowerAndUpperBounds(accuracy, y_len)
                        currFile.write(str(i) + " Accuracy:" + str(accuracy) + " Lower Bound:" +  str(lowerBounds) + " Upper Bound:" + str(upperBounds) + "\n") 

                        currFile.write("Loss calculated for layer count:" + str(layerCount) + " node count: " + str(nodeCount) +" stepSize:" + str(currStep)+" Momentum: True\n")
                        neuralNetwork = NeuralNetworkModel.NeuralNetworkModel(layerCount, nodeCount, len(xTrain[0]), addMomentum=True)
                        neuralNetwork.nodeWeight = initialWeights
                        yTestPredicted = neuralNetwork.fit(xTrain, yTrain, xTest, yTest, iterations=501, step=currStep, results=True, outputFile=currFile)
                        accuracy = Evaluations.Accuracy(yTest, yTestPredicted)
                        y_len =  len(yTestPredicted)
                        (lowerBounds, upperBounds) = ErrorBounds.Get95LowerAndUpperBounds(accuracy, y_len)
                        currFile.write(str(i) + " Accuracy:" + str(accuracy) + " Lower Bound:" +  str(lowerBounds) + " Upper Bound:" + str(upperBounds) + "\n") 




#wNode0 = neuralNetwork.nodeWeight[0][0][1:]
#wNode1 = neuralNetwork.nodeWeight[0][1][1:]

"""imgNode0 = []
imgNode1 = []

for i in range(12):
    currRow = []
    for j in range(12):
        print("at ",(12*i + j))
        currRow.append(255.0 * abs(wNode0[12*i + j]))
    imgNode0.append(currRow)

for i in range(12):
    currRow = []
    for j in range(12):
        print("at ",(12*i + j))
        currRow.append(255.0 * abs(wNode1[12*i + j]))
    imgNode1.append(currRow)
"""
"""
imgNode0 = np.multiply(255, np.abs(np.reshape(np.array(wNode0), (12, 12))))
imgNode1 = np.multiply(255, np.abs(np.reshape(np.array(wNode1), (12, 12))))

##### for visualizing in 2d
#for i in range(500):
#    print("%f, %f, %d" % (xTrain[i][0], xTrain[i][1], yTrain[i]))

##### sample image debugging output

import PIL
from PIL import Image

im0 = Image.fromarray(np.asarray(imgNode0))
if im0.mode != 'RGB':
    im0 = im0.convert('RGB')
im0.save("node0weights.jpg")

im1 = Image.fromarray(imgNode1)
if im1.mode != 'RGB':
    im1 = im1.convert('RGB')
im1.save("node1weights.jpg")

#i = Image.open(xTrainRaw[1])
#i.save("..\\..\\..\\Datasets\\FaceData\\test.jpg")

#print(i.format, i.size)

# Sobel operator
#xEdges = Assignment5Support.Convolution3x3(i, [[1, 2, 1],[0,0,0],[-1,-2,-1]])
#yEdges = Assignment5Support.Convolution3x3(i, [[1, 0, -1],[2,0,-2],[1,0,-1]])

#pixels = i.load()

#for x in range(i.size[0]):
#    for y in range(i.size[1]):
#        pixels[x,y] = abs(xEdges[x][y])

#i.save("c:\\Users\\ghult\\Desktop\\testEdgesY.jpg")"""