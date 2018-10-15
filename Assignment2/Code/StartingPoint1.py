
import Assignment1Support
import EvaluationsStub

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

#print("Logistic regression model - to")
#for i in [50000]:
#    model.fit(xTrain, yTrain, iterations=i, step=0.01)
#    yTestPredicted = model.predict(xTest)
    
#    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(xTest, yTest), EvaluationsStub.Accuracy(yTest, yTestPredicted)))
#    EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

#############################
import BagOfWordsModel
model = BagOfWordsModel.BagOfWordsModel()

xTrainRaw = []
yTrainRaw = []
xTrainRaw.append("this me")
xTrainRaw.append("that you")
xTrainRaw.append("this her")
yTrainRaw.append(1)
yTrainRaw.append(0)
yTrainRaw.append(1)
xTestRaw = []
yTestRaw = []

model.fillVocabulary(xTrainRaw)
#model.FrequencyFeatureSelection(xTestRaw, 10)
model.MutualInformationFeatureSelection(xTestRaw, yTestRaw, 10)