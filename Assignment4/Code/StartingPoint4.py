
import AddNoise
import Assignment1Support
import BagOfWordsModel
import CrossValidationSupport
import DecisionTreeModel
import EvaluationsStub
import LogisticRegressionModel
import numpy as np
import time

UseLogisticRegression = True
UseDecisionTree = True
UseForest = False
UseHandcraftedFeatures = True
MakeProblemHarder = True
UseCrossValidation = False

#frequencyValues = [0]
frequencyValues = [20, 40, 60, 80, 100]
#frequencyValues = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150] 
#mutualInformationValues = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

mutualInformationValues = [33, 66, 100, 133, 166, 200, 233, 266, 300, 333, 366, 400, 433, 466, 500]
#mutualInformationValues = [200]
#frequencyValues = [40]
treeCnts = [20]

worstCnt = 30

predictionThreshold = 0.5

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

if MakeProblemHarder == True:
    (xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
    (xTrainRaw, yTrainRaw) = AddNoise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)
    (xTestRaw, yTestRaw) = AddNoise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)
else:
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

bagModel = BagOfWordsModel.BagOfWordsModel()
bagModel.fillVocabulary(xTrainRaw)
bagModel.LoadFrequencyDictionary(xTrainRaw)
bagModel.LoadMutualInformationDictionary(xTrainRaw, yTrainRaw)

k = 1

for frequentWordCount in frequencyValues:
    for mutualInfo in mutualInformationValues:
        (xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, yTrainRaw, xTestRaw, bagModel, numFrequentWords=frequentWordCount, numMutualInformationWords=mutualInfo, includeHandCraftedFeatures=UseHandcraftedFeatures)
        yTrain = yTrainRaw
        yTest = yTestRaw

        for i in range(k):

            #print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
            #print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))
            if UseCrossValidation == True:
                print("Cross validation on ",i)
                (xTrainOnRaw, yTrainOnRaw, xValidateOnRaw, yValidateOnRaw) = CrossValidationSupport.DefineDataBounds(xTrainRaw, yTrainRaw, k, i)
                (xTrain, xValidate) = Assignment1Support.Featurize(xTrainOnRaw, yTrainOnRaw, xValidateOnRaw, bagModel, numFrequentWords=frequentWordCount, numMutualInformationWords=mutualInfo, includeHandCraftedFeatures=UseHandcraftedFeatures)

                np_xTrain = np.asarray(xTrain)
                np_xTest = np.asarray(xValidate)

                np_yTrain = np.asarray(yTrainOnRaw)
                np_yTest = np.asarray(yValidateOnRaw)
            else:  
                np_xTrain = np.asarray(xTrain)
                np_xTest = np.asarray(xTest)
                np_yTrain = np.asarray(yTrain)
                np_yTest = np.asarray(yTest)

            if UseLogisticRegression == True:
                print("Use Logistic Regression:")
                model = LogisticRegressionModel.LogisticRegressionModel()

                np_xTrainLog = np.insert(np_xTrain, 0, 1, axis = 1)
                np_xTestLog = np.insert(np_xTest, 0, 1, axis = 1)

                for i in [50000]:
                    model.fit(np_xTrainLog, np_yTrain, .5, iterations=i, step=0.01)
                    yTestPredicted = model.predict(np_xTestLog)

                    #print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), EvaluationsStub.Accuracy(np_yTest, yTestPredicted)))
                    EvaluationsStub.ConfusionMatrix(np_yTest, yTestPredicted)
                    """
                    falsePositives = model.predictWorstFPSigmoidsValues(np_xTestLog, np_yTest, worstCnt)
                    print("False Positive Raw")
                    print(falsePositives)

                    print("False Positive Results:")
                    FPresults = []
                    for j in range(worstCnt):
                        curr = falsePositives[j][0]
                        print(curr,":",yTestRaw[curr],"-",falsePositives[j][1],"=",xTestRaw[curr])
                        FPresults.append(xTestRaw[curr])

                    #worst false negatives
                    falseNegatives = model.predictWorstFNSigmoidsValues(np_xTestLog, np_yTest, worstCnt)
                    print("False Negative Raw")
                    print(falseNegatives)

                    print("False Negative Results:")   
                    FNresults = []
                    for j in range(worstCnt):
                        curr = falseNegatives[j][0]
                        print(curr,":",yTestRaw[curr],"-",falseNegatives[j][1],"=",xTestRaw[curr])
                        FPresults.append(xTestRaw[curr])
                    """
            if UseDecisionTree == True:
                print("Use Decision Tree:")
                #splitSet = [10, 25, 50, 75, 100, 125, 150, 175, 200, 275, 300, 375, 400]
                splitSet = [50]
                for currSplit in splitSet:
                    bitmap = np.ones(len(np_xTrain[0]))
                    decisionTree = DecisionTreeModel.DecisionTreeModel()
                    decisionTree.fit(np_xTrain, np_yTrain, currSplit, bitmap)
                    #print("Minimum Split:", currSplit)
                    yTestPredicted = decisionTree.predict(np_xTest, predictionThreshold) 

                    EvaluationsStub.ConfusionMatrix(np_yTest, yTestPredicted)
                    #EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
                    accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
                    y_len =  len(yTestPredicted)
                    (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
                    print(predictionThreshold,"-",EvaluationsStub.FalsePositiveRate(np_yTest, yTestPredicted),"-",EvaluationsStub.FalseNegativeRate(np_yTest, yTestPredicted))
                    print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
                #ecisionTree.PrintTree(predictionThreshold)

        #############################

            if UseForest == True:
                decisionForest = DecisionTreeModel.DecisionForestModel()

                for currTreeCnt in treeCnts:
                    print("Use Forest with tree count of ",currTreeCnt,":")
                    decisionForest.growForest(xTrain, yTrain, currTreeCnt, 2, True, 0, None)
                    treeCnt = 1
                    for currTree in decisionForest.forest:
                        yTestPredicted = currTree.predict(np_xTest, predictionThreshold) 
                        accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
                        y_len =  len(yTestPredicted)
                        (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
                        #print("Tree ",treeCnt," Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
                        treeCnt += 1
                    yTestPredicted = decisionForest.predictForest(np_xTest, predictionThreshold) 
                    accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
                    y_len =  len(yTestPredicted)
                    (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
                    print("Total Forest Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
