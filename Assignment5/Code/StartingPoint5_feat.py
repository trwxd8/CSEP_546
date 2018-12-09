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

"""from PIL import Image
import Assignment5Support

image = Image.open('Oscar_DLeon_0001_R.jpg')

yGradients = np.array(Assignment5Support.Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]))
xGradients = np.array(Assignment5Support.Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))

# y-gradient 9 grids of 8x8 pixels
yFeatures = Assignment5Support.CalculateGradientFeatures(yGradients)
print (yFeatures[:5])

# x-gradient 9 grids of 8x8 pixels
xFeatures = Assignment5Support.CalculateGradientFeatures(xGradients)
print(xFeatures[:5])

# y-graident 5-bin histogram
yFeatures = Assignment5Support.CalculateHistogramFeatures(yGradients)
print (yFeatures[:5])

# x-gradient 5-bin histogram
xFeatures = Assignment5Support.CalculateHistogramFeatures(xGradients)
print (xFeatures[:5])
"""
print("Calculating features... XGradients")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeXGradients=False, includeYGradients=True, includeYHistogram=False, includeXHistogram=False, includeRawPixels=False, includeIntensities=False)
yTrain = yTrainRaw
yTest = yTestRaw


import Evaluations
import ErrorBounds

######
import MostCommonModel
model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)
print("Most Common Accuracy:", Evaluations.Accuracy(yTest, yTestPredicted), ErrorBounds.Get95LowerAndUpperBounds(Evaluations.Accuracy(yTest, yTestPredicted), len(yTest)))

######
import DecisionTreeModel
decisionForest = DecisionTreeModel.DecisionForest()
#MinsToSplit = [60]
#treeCnts = [25]
#featureRestriction = [30]

MinsToSplit = [90]
treeCnts = [40]
featureRestriction = [20]

#MinsToSplit = [30, 60, 90, 120, 150, 180]
#treeCnts = [40, 50, 60, 70]
#featureRestriction = [ 3, 4]

np_xTrain = np.array(xTrain)
np_yTrain = np.array(yTrain)
np_xTest = np.array(xTest)
np_yTest = np.array(yTest)

for currRestriction in featureRestriction:
    for currTreeCnt in treeCnts:
            for currSplit in MinsToSplit:
                #x, y, numTrees, minSplit, baggingFlag, featureRestriction, seed
                print("Restriction:",currRestriction," Tree count:",currTreeCnt," minsplit:",currSplit)
                decisionForest.growForest(np_xTrain, np_yTrain, currTreeCnt, currSplit, True, currRestriction, None)
                #treeCnt = 1
                #for currTree in decisionForest.forest:
                #        yTestPredicted = currTree.predict(np_xTest, predictionThreshold) 
                #        accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
                #        y_len =  len(yTestPredicted)
                #        (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
                        #print("Tree ",treeCnt," Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
                #        treeCnt += 1
                yTestPredicted = decisionForest.predictForest(np_xTest, .5)
                accuracy = Evaluations.Accuracy(np_yTest, yTestPredicted)
                y_len =  len(yTestPredicted)
                (lowerBounds, upperBounds) = ErrorBounds.Get95LowerAndUpperBounds(accuracy, y_len)
                print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds) 
                for predictionThreshold in thresholds:
                    yTestPredicted = decisionForest.predictForest(np_xTest, predictionThreshold)

                    print(predictionThreshold,"-",Evaluations.FalsePositiveRate(np_yTest, yTestPredicted),"-",Evaluations.FalseNegativeRate(np_yTest, yTestPredicted))

                #accuracy = Evaluations.Accuracy(np_yTest, yTestPredicted)
                #y_len =  len(yTestPredicted)
                #(lowerBounds, upperBounds) = ErrorBounds.Get95LowerAndUpperBounds(accuracy, y_len)
                #print("Total Forest Accuracy for feature restriction of ",currRestriction,":", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)

"""
model = DecisionTreeModel.DecisionTree()
bitmap = np.ones(len(xTrain[0]))

for currSplit in minToSplits:
    model.fit(xTrain, yTrain, currSplit, bitmap)
    yTestPredicted = model.predict(xTest, predictionThreshold)
    print("Decision Tree Accuracy at ",currSplit,":", Evaluations.Accuracy(yTest, yTestPredicted), ErrorBounds.Get95LowerAndUpperBounds(Evaluations.Accuracy(yTest, yTestPredicted), len(yTest)))

##### for visualizing in 2d
#for i in range(500):
#    print("%f, %f, %d" % (xTrain[i][0], xTrain[i][1], yTrain[i]))

##### sample image debugging output

import PIL
from PIL import Image

i = Image.open(xTrainRaw[1])
#i.save("..\\..\\..\\Datasets\\FaceData\\test.jpg")

print(i.format, i.size)

# Sobel operator
xEdges = Assignment5Support.Convolution3x3(i, [[1, 2, 1],[0,0,0],[-1,-2,-1]])
yEdges = Assignment5Support.Convolution3x3(i, [[1, 0, -1],[2,0,-2],[1,0,-1]])

pixels = i.load()

for x in range(i.size[0]):
    for y in range(i.size[1]):
        pixels[x,y] = abs(xEdges[x][y])

#i.save("c:\\Users\\ghult\\Desktop\\testEdgesY.jpg")"""