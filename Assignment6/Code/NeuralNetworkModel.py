import numpy as np
import copy

class NeuralNetworkModel(object):
    """A model that predicts the most common label from the training data."""

    def __init__(self, hiddenLayerCount, hiddenLayerSize, inputLen, addMomentum=False):
        self.layerCount = hiddenLayerCount
        self.layerSize = hiddenLayerSize
        self.nodeWeight = []
        self.nodeActivations = []
        self.momentum = addMomentum
        if addMomentum:
            self.momentumWeight = np.random.ranf()

        #Go through each layer 
        for i in range(hiddenLayerCount):

            #Determine how many weight values to initialize
            # +1 for x_0 = 1
            if i == 0:
                weightCnt = inputLen + 1
            else:
                weightCnt = hiddenLayerSize + 1
            currLayer = []
            currActivations = []
            #currDeltas = []

            #For each node append blank value for activation value and randoms weights initialized between -.05 to .05
            for j in range(hiddenLayerSize):
                currActivations.append(0)
                currLayer.append(np.multiply(np.subtract(np.random.ranf(weightCnt), .5), .1))
                #currDeltas.append(np.zeros((weightCnt)))
            #Append node set to collection
            self.nodeWeight.append(currLayer)
            self.nodeActivations.append(currActivations)

        #Append extra weights and final activation for result layer
        self.nodeWeight.append([np.multiply(np.subtract(np.random.ranf(hiddenLayerSize + 1), .5), .1)])
        #currDeltas.append([np.zeros((hiddenLayerSize + 1))])
        self.nodeActivations.append([0])
        self.blankDeltas = copy.deepcopy(self.nodeActivations)
        #self.blankDeltas = copy.deepcopy(currDeltas)
        #print(self.nodeWeight)
        #print(self.nodeActivations)
        #input("Baselinee")       


    def fit(self, xTrain, yTrain, xTest, yTest, iterations, step, results, outputFile):
        trainLen = len(xTrain)
        testLen = len(xTest)

        self.sampleDeltaWeights = []
        #Duplicate weight shape to hold delta weights for momentum
        #Set here to re-initialize first weights to 0 
        for i in range(trainLen):
            self.sampleDeltaWeights.append(copy.deepcopy(self.blankDeltas))
        
        prevLoss = 1
        for curr_iter in range(iterations):
            #print("Iter ",curr_iter)
            yPredictions = []
            for i in range(trainLen):
                sample = xTrain[i]
                answer = yTrain[i]
                #print("Running model for sample ",i)
                prediction = self.ForwardPropogation(sample)
                yPredictions.append(prediction)
                self.BackwardsPropogation(prediction, sample, answer, i, step)
            currLoss = self.loss(yPredictions, yTrain)
            if curr_iter%5 == 0 or currLoss > prevLoss:
                outputFile.write("train" + str(curr_iter) + " " + str(currLoss) + " greater than " + str(prevLoss) + "\n")
                yTestPredictions = []
                for i in range(testLen):
                    sample = xTest[i]
                    prediction = self.ForwardPropogation(sample)
                    yTestPredictions.append(prediction)
                currTestLoss = self.loss(yTestPredictions, yTest)
                outputFile.write("test" + str(curr_iter) + " " + str(currTestLoss) + "\n")
            if currLoss <= prevLoss:
                prevLoss = currLoss
            else:
                break
        yTestPredictions = []
        for i in range(testLen):
            sample = xTest[i]
            prediction = self.ForwardPropogation(sample)
            yTestPredictions.append(prediction)
        return self.makePrediction(yTestPredictions, .5)

    def makePrediction(self, predictions, threshold):
        yPredictions = []
        for sample in predictions:
            #print("Predicting ",sample)
            yPredictions.append( 1 if sample > threshold else 0)
        return yPredictions


    def ForwardPropogation(self, sample):
        #Make sample 1D?
        nodeInputs = np.insert(np.asarray(sample), 0, 1)
        for layer in range(self.layerCount):
            #print(nodeInputs)
            #print(self.nodeWeight)
            newInputs = [1]
            #print("Layer ",layer,":")
            for node in range(self.layerSize):
                nodeValue = self.calculateSigmoids(nodeInputs, self.nodeWeight[layer][node])
                #print("Node (",layer,",",node,"): ",nodeValue)
                self.nodeActivations[layer][node] = nodeValue
                newInputs.append(nodeValue)
            nodeInputs = newInputs
        
        nodeValue = self.calculateSigmoids(nodeInputs, self.nodeWeight[self.layerCount][0])
        self.nodeActivations[self.layerCount] = [nodeValue]
        #print("Prediction:",nodeValue)
        return nodeValue

    def calculateSigmoids(self, x, weights):
        #print("X:", x)
        #print("weights:", weights)
        return np.divide(1.0, np.add(1.0, np.exp(np.multiply(-1.0, np.dot(x, weights)))))

    def BackwardsPropogation(self, prediction, sample, answer, sampleIdx, step):
        #currErrors = []

        sampleDeltaWeights = self.sampleDeltaWeights[sampleIdx]

        #print("starting deltas:",sampleDeltaWeights)
        currErrors = []
        currWeights = []
        for currLayer in range(self.layerCount, -1, -1):
            #print("At layer ", currLayer)
            prevErrors = []
            prevWeights = []
            #print(self.nodeActivations[currLayer])
            nodeCount = len(self.nodeActivations[currLayer])
            if currLayer == 0:
                nodeInputs = np.insert(np.asarray(sample), 0, 1)
                #nodeInputs = np.asarray(sample)
            else:
                nodeInputs = np.insert(np.asarray(self.nodeActivations[currLayer-1]), 0, 1)
                #nodeInputs = np.asarray(self.nodeActivations[currLayer-1])
            #print("Node Inputs:", nodeInputs)
            for currNode in range(nodeCount):
                node = self.nodeActivations[currLayer][currNode]
                if(nodeCount == 1):
                    error = self.TotalErrorFunction(node, answer)
                else:
                    error = self.NodeErrorFunction(node, currNode, currWeights, currErrors)
                #print("Node (",currLayer,",",currNode,"): ",error)
                #print("weight:",self.nodeWeight[currLayer][currNode])
                if self.momentum:
                    weightAdjustments = np.add(np.multiply(error, np.multiply(step, nodeInputs)), np.multiply(self.momentumWeight, sampleDeltaWeights[currLayer][currNode]))
                    #print("Weight adjustments (",currLayer,",",currNode,"): ", weightAdjustments)
                    sampleDeltaWeights[currLayer][currNode] = weightAdjustments
                else:
                    weightAdjustments = np.multiply(error, np.multiply(step, nodeInputs))
                #print("Weights Adjustment:",weightAdjustments)
                #print("Weights:", self.nodeWeight[currLayer][currNode])
                prevWeights.append(self.nodeWeight[currLayer][currNode])
                self.nodeWeight[currLayer][currNode] = np.add(self.nodeWeight[currLayer][currNode], weightAdjustments)
                #print("weight now:",self.nodeWeight[currLayer][currNode])

                prevErrors.append(error)
            currErrors = prevErrors
            currWeights = prevWeights

    def TotalErrorFunction(self, prediction, answer):
        return prediction * (1 - prediction) * (answer - prediction)

    def NodeErrorFunction(self, prediction, nodeIdx, weights, errors):
        totalError = prediction * (1 - prediction)
        for i in range(len(errors)):
            #print("error:", errors[i])
            #print("weights:", weights[i][nodeIdx+1])
            totalError = totalError * (weights[i][nodeIdx+1]*errors[i])
        return totalError

    def loss(self, yPredicted, y):
        count = len(y)
        return np.divide(np.sum(np.divide(np.square(np.subtract(yPredicted, y)), 2)), count)

    def UnitTest(self):
        self.layerCount = 1
        self.layerSize = 2
        self.nodeWeight = np.array([[[.5, -1.0, 1.0], [1.0, 0.5, -1.0]],
                           [[.25, 1.0, 1.0]]])
        self.nodeActivations = np.array([[0, 0,], [0]])
        self.blankDeltas = np.array([[[0, 0, 0], [0, 0, -0]],
                           [[0, 0, 0]]])

        print(self.nodeWeight)
        print(self.nodeActivations)
        print(self.blankDeltas)
        input("Baselinee")       

        training = []
        answers = []
        sample = np.array([1.0, 0.5])
        training.append(sample)
        answers.append(1)
        self.fit(training, answers, [0,0],[0], 3, .1, False)
        
    def UnitTestTwo(self):
        self.layerCount = 2
        self.layerSize = 2
        self.nodeWeight = np.array([[[.5, .75, .2, 1.0], [.3, .8, 0.25, .5]],
                                    [[.6, .25, .9], [1, .1, .5]], 
                                    [[.75, 1, .5]]])
        self.nodeActivations = np.array([[0, 0], [0, 0], [0]])
        self.blankDeltas = copy.deepcopy(self.nodeActivations)

        print(self.nodeWeight.shape)
        print(self.nodeActivations)
        input("Baselinee")       

        training = []
        answers = []
        sample = np.array([1.0, .75, 0.4])
        training.append(sample)
        answers.append(1)
        #self.fit(training, answers, [0,0],[0], 1, .05)