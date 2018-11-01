
import AddNoise
import Assignment1Support
import DecisionTreeModel
import EvaluationsStub
import numpy as np
import time

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

#(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
#(xTrain, xTest) = Assignment1Support.FeaturizeHandcrafted(xTrainRaw, xTestRaw)
#yTrain = yTrainRaw
#yTest = yTestRaw


(xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
(xTrainRaw, yTrainRaw) = AddNoise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)
(xTestRaw, yTestRaw) = AddNoise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, yTrainRaw, xTestRaw, numFrequentWords=0, numMutualInformationWords=295, includeHandCraftedFeatures=True)
yTrain = yTrainRaw
yTest = yTestRaw

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))


np_xTrain = np.asarray(xTrain)
np_xTest = np.asarray(xTest)
np_yTrain = np.asarray(yTrain)
np_yTest = np.asarray(yTest)
"""
thresholds = [.5]
#for i in range(0,101):
#    thresholds.append(i*.01)
currSplit = 65
bitmap = np.ones(len(np_xTrain[0]))
print(bitmap)
decisionTree = DecisionTreeModel.DecisionTreeModel()
decisionTree.fit(np_xTrain, np_yTrain, currSplit, bitmap)
for currThreshold in thresholds:   
    #print("Minimum Split:", currSplit)
    yTestPredicted = decisionTree.predict(np_xTest, currThreshold) 
    #EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
    accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
    y_len =  len(yTestPredicted)
    (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
    print(currThreshold,"-",EvaluationsStub.FalsePositiveRate(np_yTest, yTestPredicted),"-",EvaluationsStub.FalseNegativeRate(np_yTest, yTestPredicted))
    #print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
    decisionTree.PrintTree(currThreshold)

#############################
import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

#np_xTrain = np.insert(np_xTrain, 0, 1, axis = 1)
#np_xTest = np.insert(np_xTest, 0, 1, axis = 1)

for i in [50000]:
    model.fit(np_xTrain, np_yTrain, .5, iterations=i, step=0.01)
    yTestPredicted = model.predict(np_xTest)

    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), EvaluationsStub.Accuracy(np_yTest, yTestPredicted)))
    EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)

#############################
"""
decisionForest = DecisionTreeModel.DecisionForestModel()

predictionThreshold = 0.5

treeCnts = [1, 20, 40, 60, 80]
for currTreeCnt in treeCnts:
    decisionForest.growForest(xTrain, yTrain, currTreeCnt, 2, True, 0, None)
    treeCnt = 1
    for currTree in decisionForest.forest:
        yTestPredicted = currTree.predict(np_xTest, predictionThreshold) 
        accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
        y_len =  len(yTestPredicted)
        (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
        print("Tree ",treeCnt," Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
        treeCnt += 1
    yTestPredicted = decisionForest.predictForest(np_xTest, predictionThreshold) 
    accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
    y_len =  len(yTestPredicted)
    (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
    print("Total Forest Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
