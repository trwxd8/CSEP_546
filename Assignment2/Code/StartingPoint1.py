
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
bagmodel = BagOfWordsModel.BagOfWordsModel()

print("Running Bag of Words")
topN = 10

bagmodel.fillVocabulary(xTrainRaw)
results = bagmodel.FrequencyFeatureSelection(xTestRaw, topN)
#results = bagmodel.MutualInformationFeatureSelection(xTrainRaw, yTrainRaw, 10)

print(results)

words = []
for i in range(topN):
    words.append(results[i][0])

print("Logistic regression model - Using Bag of model (frequency")
xTrain, xTest = bagmodel.FeaturizeByWords(xTrainRaw, xTestRaw, words)

for i in [50000]:
    model.fit(xTrain, yTrain, iterations=i, step=0.01)
    yTestPredicted = model.predict(xTest)
    
    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(xTest, yTest), EvaluationsStub.Accuracy(yTest, yTestPredicted)))
    EvaluationsStub.ExecuteAll(yTest, yTestPredicted)