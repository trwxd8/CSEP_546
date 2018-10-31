
import AddNoise
import Assignment1Support
import EvaluationsStub
import numpy as np
import time

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
(xTrainRaw, yTrainRaw) = AddNoise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)
(xTestRaw, yTestRaw) = AddNoise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, yTrainRaw, xTestRaw, numFrequentWords=0, numMutualInformationWords=9, includeHandCraftedFeatures=True)
yTrain = yTrainRaw
yTest = yTestRaw

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

print(xTest)

np_xTrain = np.asarray(xTrain)
np_xTest = np.asarray(xTest)
np_yTrain = np.asarray(yTrain)
np_yTest = np.asarray(yTest)

print(np_xTrain.shape)
print(np_xTest.shape)

np_xTrain = np.insert(np_xTrain, 0, 1, axis = 1)
np_xTest = np.insert(np_xTest, 0, 1, axis = 1)


import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

for i in [50000]:
    model.fit(np_xTrain, np_yTrain, .5, iterations=i, step=0.01)
    yTestPredicted = model.predict(np_xTest)

    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), EvaluationsStub.Accuracy(np_yTest, yTestPredicted)))
    EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
"""
#############################
import DecisionTreeModel
#decisionTree = DecisionTreeModel.DecisionTreeUnitTest()
#decisionTree.ExecuteTest()
decisionTree = DecisionTreeModel.DecisionTreeModel()

#minSplits = [100]
currSplit = 65

np_xTrain = np.asarray(xTrain)
np_xTest = np.asarray(xTest)

np_yTrain = np.asarray(yTrain)
np_yTest = np.asarray(yTest)

thresholds = [.5]
#for i in range(0,101):
#    thresholds.append(i*.01)

decisionTree.fit(np_xTrain, np_yTrain, currSplit, 2, 4)
for currThreshold in thresholds:   
    #print("Minimum Split:", currSplit)
    yTestPredicted = decisionTree.predict(np_xTest, currThreshold) 
    #EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
    accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
    y_len =  len(yTestPredicted)
    (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
    #print(currThreshold,"-",EvaluationsStub.FalsePositiveRate(np_yTest, yTestPredicted),"-",EvaluationsStub.FalseNegativeRate(np_yTest, yTestPredicted))
    print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
#decisionTree.PrintTree(currThreshold)
"""