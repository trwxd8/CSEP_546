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
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeXGradients=False, includeYGradients=False, includeYHistogram=False, includeXHistogram=False, includeRawPixels=False, includeIntensities=False)
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
"""
import DecisionTreeModel
decisionForest = DecisionTreeModel.DecisionForest()

MinsToSplit = [90]
treeCnts = [40]
featureRestriction = [20]

for currRestriction in featureRestriction:
    for currTreeCnt in treeCnts:
            for currSplit in MinsToSplit:
                #x, y, numTrees, minSplit, baggingFlag, featureRestriction, seed
                print("Restriction:",currRestriction," Tree count:",currTreeCnt," minsplit:",currSplit)
                decisionForest.growForest(np_xTrain, np_yTrain, currTreeCnt, currSplit, True, currRestriction, None)
                yTestPredicted = decisionForest.predictForest(np_xTest, .5)
                accuracy = Evaluations.Accuracy(np_yTest, yTestPredicted)
                y_len =  len(yTestPredicted)
                (lowerBounds, upperBounds) = ErrorBounds.Get95LowerAndUpperBounds(accuracy, y_len)
                print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds) 
                #for predictionThreshold in thresholds:
                #    yTestPredicted = decisionForest.predictForest(np_xTest, predictionThreshold)
                #    print(predictionThreshold,"-",Evaluations.FalsePositiveRate(np_yTest, yTestPredicted),"-",Evaluations.FalseNegativeRate(np_yTest, yTestPredicted))
"""
######
import KClustering

iterations = 10
k = 4
len(xTrain)
np_xTrainNormalized = np.divide(np_xTrain, 255.0)
#for i in range(len(np_xTrainNormalized)):
#    print(np_xTrainNormalized[i][0]," ",np_xTrainNormalized[i][1])
clusters = KClustering.KMeansClustering(k, 4)
clusters.fitCentroids(np_xTrainNormalized)
#clusters.FindClosestImage(np_xTrainNormalized, xTrainRaw)
#clusters.formCluster(iterations, np_xTrainNormalized)


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