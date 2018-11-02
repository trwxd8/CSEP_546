
import AddNoise
import Assignment1Support
import BagOfWordsModel
import DecisionTreeModel
import EvaluationsStub
import LogisticRegressionModel
import numpy as np
import time

UseLogisticRegression = True
UseDecisionTree = False
UseForest = False
UseHandcraftedFeatures = True
MakeProblemHarder = False


frequencyValues = [0]
mutualInformationValues = [0, 10, 15, 20, 30, 50]
treeCnts = [1, 20, 40, 60, 80]

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

for frequentWordCount in frequencyValues:
    for mutualInfo in mutualInformationValues:
        (xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, yTrainRaw, xTestRaw, numFrequentWords=frequentWordCount, numMutualInformationWords=mutualInfo, includeHandCraftedFeatures=UseHandcraftedFeatures)
        yTrain = yTrainRaw
        yTest = yTestRaw

        #print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
        #print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

        np_xTrain = np.asarray(xTrain)
        np_xTest = np.asarray(xTest)
        np_yTrain = np.asarray(yTrain)
        np_yTest = np.asarray(yTest)

        if UseLogisticRegression == True:
            print("Use Logistic Regression:")
            model = LogisticRegressionModel.LogisticRegressionModel()

            np_xTrain = np.insert(np_xTrain, 0, 1, axis = 1)
            np_xTest = np.insert(np_xTest, 0, 1, axis = 1)

            for i in [50000]:
                model.fit(np_xTrain, np_yTrain, .5, iterations=i, step=0.01)
                yTestPredicted = model.predict(np_xTest)

                #print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(np_xTest, np_yTest), EvaluationsStub.Accuracy(np_yTest, yTestPredicted)))
                EvaluationsStub.ConfusionMatrix(np_yTest, yTestPredicted)

        if UseDecisionTree == True:
            print("Use Decision Tree:")
            currSplit = 65
            bitmap = np.ones(len(np_xTrain[0]))
            print(bitmap)
            decisionTree = DecisionTreeModel.DecisionTreeModel()
            decisionTree.fit(np_xTrain, np_yTrain, currSplit, bitmap)
            #print("Minimum Split:", currSplit)
            yTestPredicted = decisionTree.predict(np_xTest, predictionThreshold) 
            #EvaluationsStub.ExecuteAll(np_yTest, yTestPredicted)
            accuracy = EvaluationsStub.Accuracy(np_yTest, yTestPredicted)
            y_len =  len(yTestPredicted)
            (lowerBounds, upperBounds) = EvaluationsStub.calculate95PercentConfidenceBounds(accuracy, y_len)
            print(predictionThreshold,"-",EvaluationsStub.FalsePositiveRate(np_yTest, yTestPredicted),"-",EvaluationsStub.FalseNegativeRate(np_yTest, yTestPredicted))
            #print("Accuracy:", accuracy, " Lower Bound:", lowerBounds, " Upper Bound:", upperBounds)
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
