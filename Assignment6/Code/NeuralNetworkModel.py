import numpy as np

class NeuralNetworkModel(object):
    """A model that predicts the most common label from the training data."""

    def __init__(self, hiddenLayerCount, hiddenLayerSize, inputLen):
        self.layerCount = hiddenLayerCount
        self.layerSize = hiddenLayerSize
        self.nodeWeight = []
        self.nodeActivations = []

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

            #For each node append blank value for activation value and randoms weights initialized between -.05 to .05
            for j in range(hiddenLayerSize):
                currActivations.append(0)
                currLayer.append(np.multiply(np.subtract(np.random.ranf(weightCnt), .5), .1))
            #Append node set to collection
            self.nodeWeight.append(currLayer)
            self.nodeActivations.append(currActivations)

        #Append extra weights and final activation for result layer
        self.nodeWeight.append([np.multiply(np.subtract(np.random.ranf(weightCnt), .5), .1)])
        self.nodeActivations.append([0])

        #print(self.nodeWeight)
        #print(self.nodeActivations)
        #input("Baselinee")

    def fit(self, xTrain, yTrain, iterations, step):
        trainLen = len(xTrain)
        inputLen = len(xTrain[0])
        for curr_iter in range(iterations):
            for i in range(trainLen):
                sample = xTrain[i]
                answer = yTrain[i]
                #print("Running model for sample ",i)
                prediction = self.ForwardPropogation(sample)
                #error =  np.abs(prediction - answer)
                self.BackwardsPropogation(prediction, sample, answer, step)
        input("Run complete")

        pass

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
        return np.divide(1.0, np.add(1.0, np.exp(np.multiply(-1.0, np.dot(x, weights)))))

    def BackwardsPropogation(self, prediction, sample, answer, step):
        #currErrors = []

        #error =  self.ErrorFunction(prediction, answer)
        #print("Starting nodes:", self.nodeActivations)
        #currErrors.append(error)
        #print("Node activations:",self.nodeActivations)
        #print("weights:",self.nodeWeight)
        currErrors = []
        currWeights = []
        for currLayer in range(self.layerCount, -1, -1):
            prevErrors = []
            prevWeights = []
            #print(self.nodeActivations[currLayer])
            nodeCount = len(self.nodeActivations[currLayer])
            if currLayer == 0:
                #weightCnt = len(sample) + 1
                nodeInputs = np.insert(np.asarray(sample), 0, 1)
            else:
                #weightCnt = self.layerSize + 1
                nodeInputs = np.insert(np.asarray(self.nodeActivations[currLayer-1]), 0, 1)
            for currNode in range(nodeCount):
                node = self.nodeActivations[currLayer][currNode]
                if(nodeCount == 1):
                    error = self.TotalErrorFunction(node, answer)
                else:
                    error = self.NodeErrorFunction(node, currWeights, currErrors)
                #print("Node (",currLayer,",",currNode,"): ",error)
                weightAdjustments = np.multiply(error, np.multiply(step, nodeInputs))
                prevWeights.append(self.nodeWeight[currLayer][currNode][currNode+1])
                self.nodeWeight[currLayer][currNode] = np.add(self.nodeWeight[currLayer][currNode], weightAdjustments)
                #print("weights now ", self.nodeWeight[currLayer][currNode])
                prevErrors.append(error)
            currErrors = prevErrors
            currWeights = prevWeights

    def TotalErrorFunction(self, prediction, answer):
        return prediction * (1 - prediction) * (answer - prediction)

    def NodeErrorFunction(self, prediction, weights, errors):
        totalError = prediction * (1 - prediction)
        for i in range(len(errors)):
            totalError = totalError * (weights[i]*errors[i])
        return totalError

    def UnitTest(self):
        self.layerCount = 1
        self.layerSize = 2
        self.nodeWeight = np.array([[[.5, -1.0, 1.0], [1.0, 0.5, -1.0]],
                           [[.25, 1.0, 1.0], [-1, -1, -1]]])
        training = []
        answers = []
        sample = np.array([1.0, 0.5])
        training.append(sample)
        answers.append(1)
        self.fit(training, answers, 1, .1)