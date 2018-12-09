
import Assignment1Support
import EvaluationsStub
import numpy as np
import time

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))
"""
############################
import MostCommonModel

model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

print("### 'Most Common' model")

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

############################
import SpamHeuristicModel
model = SpamHeuristicModel.SpamHeuristicModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

print("### Heuristic model")

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)
"""
#############################
(xTrain, xTest) = Assignment1Support.FeaturizeHandcrafted(xTrainRaw, xTestRaw)
#(xTrain, xTest) = Assignment1Support.FeaturizeLength(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw


#############################
import DecisionTreeModel
#decisionTree = DecisionTreeModel.DecisionTreeUnitTest()
#decisionTree.ExecuteTest()
decisionTree = DecisionTreeModel.DecisionTreeModel()

#minSplits = [100]
currSplit = 75

np_xTrain = np.asarray(xTrain)
np_xTest = np.asarray(xTest)

np_yTrain = np.asarray(yTrain)
np_yTest = np.asarray(yTest)

thresholds = [.5]
#for i in range(0,101):
#    thresholds.append(i*.01)

decisionTree.fit(np_xTrain, np_yTrain, currSplit)
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
"""import DecisionTreeModel
import CrossValidationSupport

k = 5
full_cnt = len(xTrainRaw)
minSplits = []

for i in range(39, 500):
    if i % 10 == 0:
        minSplits.append(i)

for currSplit in minSplits:   
    print("Minimum Split:", currSplit)
    for i in range(k):
        (xTrainOn, yTrainOn, xValidateOn, yValidateOn) = CrossValidationSupport.DefineDataBounds(xTrain, yTrain, k, i)

        np_xTrain = np.asarray(xTrainOn)
        np_xTest = np.asarray(xValidateOn)

        np_yTrain = np.asarray(yTrainOn)
        np_yTest = np.asarray(yValidateOn)

        decisionTree = DecisionTreeModel.DecisionTreeModel()
        decisionTree.fit(np_xTrain, np_yTrain, currSplit)
        #decisionTree.PrintTree()
        yTestPredicted = decisionTree.predict(np_xTest, .5) 

        accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
        y_len =  len(yTestPredicted)
        (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
        print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
        #decisionTree.PrintTree()"""