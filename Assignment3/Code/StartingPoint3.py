
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

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

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

############################
import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

print("Logistic regression model")

#np_xTrain = np.insert(np.asarray(xTrain), 0, 1, axis = 1)
#np_xTest = np.insert(np.asarray(xTest), 0, 1, axis = 1)
np_yTrain = np.asarray(yTrain)
np_yTest = np.asarray(yTest)

#cntList = [50000]
#for i in cntList:
#    start = time.time()
#    model.fit(np_xTrain, np_yTrain, iterations=i, step=0.01)
#    yTestPredicted = model.predict(np_xTest)
#    end = time.time()
#    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), EvaluationsStub.Accuracy(np_yTest, yTestPredicted)))
#    EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
#    print("Model runtime:", end-start)

#############################
"""import BagOfWordsModel
bagmodel = BagOfWordsModel.BagOfWordsModel()

print("Running Bag of Words")
topN = 10

bagmodel.fillVocabulary(xTrainRaw)
results = bagmodel.MutualInformationFeatureSelection(xTrainRaw, yTrainRaw, topN)

print(results)

words = []
for i in range(topN):
    words.append(results[i][0])

print("Logistic regression model - Using Bag of model")
xTrainMI, xTestMI = bagmodel.FeaturizeByWords(xTrainRaw, xTestRaw, words)

np_xTrainFullFeautures = np.hstack((np.asarray(xTrain), np.asarray(xTrainMI)))
np_xTestFullFeautures = np.hstack((np.asarray(xTest), np.asarray(xTestMI)))
#np_xTrainFullFeautures = np.asarray(xTrainMI)
#np_xTestFullFeautures = np.asarray(xTestMI)

np_xTrain = np.insert(np_xTrainFullFeautures, 0, 1, axis = 1)
np_xTest = np.insert(np_xTestFullFeautures, 0, 1, axis = 1)

thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
for currThreshold in thresholds :
    for i in [50000]:
        print("Threshold at ",currThreshold)
        model.fit(np_xTrain, np_yTrain, iterations=i, step=0.01, threshold=currThreshold)
        yTestPredicted = model.predict(np_xTest)

        accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
        y_len =  len(yTestPredicted)
        (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
        print(i,": ", accuracy, " l:", lowerBounds, " u:", upperBounds)

        print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), accuracy))
        EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
"""
#############################
import DecisionTreeModel
#decisionTree = DecisionTreeModel.DecisionTreeUnitTest()
#decisionTree.ExecuteTest()
decisionTree = DecisionTreeModel.DecisionTreeModel()

#minSplits = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
minSplits = [100]

np_xTrain = np.asarray(xTrain)
np_xTest = np.asarray(xTest)

for currSplit in minSplits:   
    print("Minimum Split:", currSplit)
    decisionTree.fit(np_xTrain, np_yTrain, currSplit)
    yTestPredicted = decisionTree.predict(np_xTest) 
    EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
    accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
    y_len =  len(yTestPredicted)
    (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
    print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)