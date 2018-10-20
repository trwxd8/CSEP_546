
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
import BagOfWordsModel
bagmodel = BagOfWordsModel.BagOfWordsModel()

print("Running Bag of Words")
topN = 10

bagmodel.fillVocabulary(xTrainRaw)
#results = bagmodel.FrequencyFeatureSelection(xTrainRaw, topN)
results = bagmodel.MutualInformationFeatureSelection(xTrainRaw, yTrainRaw, topN)

print(results)

words = []
for i in range(topN):
    words.append(results[i][0])

print("Logistic regression model - Using Bag of model")
xTrainMI, xTestMI = bagmodel.FeaturizeByWords(xTrainRaw, xTestRaw, words)

#np_xTrainFullFeautures = np.hstack((np.asarray(xTrain), np.asarray(xTrainMI)))
#np_xTestFullFeautures = np.hstack((np.asarray(xTest), np.asarray(xTestMI)))
np_xTrainFullFeautures = np.asarray(xTrainMI)
np_xTestFullFeautures = np.asarray(xTestMI)

np_xTrain = np.insert(np_xTrainFullFeautures, 0, 1, axis = 1)
np_xTest = np.insert(np_xTestFullFeautures, 0, 1, axis = 1)

for i in [50000]:
    model.fit(np_xTrain, np_yTrain, iterations=i, step=0.01)
    #yTestPredicted = model.predict(np_xTest)

    #print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), EvaluationsStub.Accuracy(np_yTest, yTestPredicted)))
    #EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)

    #accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
    #y_len =  len(yTestPredicted)
    #(lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
    #print(i,": ", accuracy, " l:", lowerBounds, " u:", upperBounds)

    #Worst false positives
    falsePositives = model.predictWorstFPSigmoidsValues(np_xTest, np_yTest, 20)
    print("False Positive Raw")
    print(falsePositives)

    print("False Positive Results:")
    FPresults = []
    for j in range(20):
        curr = falsePositives[j][0]
        print(curr,":",yTestRaw[curr],"-",falsePositives[j][1],"=",xTestRaw[curr])
        FPresults.append(xTestRaw[curr])

    #worst false negatives
    falseNegatives = model.predictWorstFNSigmoidsValues(np_xTest, np_yTest, 20)
    print("False Negative Raw")
    print(falseNegatives)

    print("False Negative Results:")   
    FNresults = []
    for j in range(20):
        curr = falseNegatives[j][0]
        print(curr,":",yTestRaw[curr],"-",falseNegatives[j][1],"=",xTestRaw[curr])
        FPresults.append(xTestRaw[curr])




"""
#############################
import CrossValidationSupport

import BagOfWordsModel
bagmodel = BagOfWordsModel.BagOfWordsModel()

topN = 10
k = 5
full_cnt = len(xTrainRaw)
correct_cnt = 0
for i in range(k):
    (xTrainOnRaw, yTrainOnRaw, xValidateOnRaw, yValidateOnRaw) = CrossValidationSupport.DefineDataBounds(xTrainRaw, yTrainRaw, k, i)

    bagmodel.fillVocabulary(xTrainOnRaw)
    results = bagmodel.FrequencyFeatureSelection(xTrainOnRaw, topN)
    #results = bagmodel.MutualInformationFeatureSelection(xTrainOnRaw, yTrainOnRaw, topN)

    words = []
    for j in range(topN):
        words.append(results[j][0])
    print(words)
    
    xTrainMI, xValidateMI = bagmodel.FeaturizeByWords(xTrainOnRaw, xValidateOnRaw, words)

    np_xTrain = np.insert(np.asarray(xTrainMI), 0, 1, axis = 1)
    np_xValidate = np.insert(np.asarray(xValidateMI), 0, 1, axis = 1)

    np_yTrain = np.asarray(yTrainOnRaw)
    np_yValidate = np.asarray(yValidateOnRaw)

    for j in [50000]:
        model.fit(np_xTrain, np_yTrain, iterations=j, step=0.01)
        yValidatePredicted = model.predict(np_xValidate)

    validateLen = len(yValidatePredicted)
    for j in range(validateLen):
        if yValidatePredicted[j] == np_yValidate[j]:
            correct_cnt += 1
    print("Correct after ",i," is ", correct_cnt)

accuracy = correct_cnt / full_cnt
(lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, full_cnt)
print("MI: ", accuracy, " l:", lowerBounds, " u:", upperBounds) """