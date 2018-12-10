import numpy as np
import torch
import torch.nn.functional as func
import math

class QLearning():
    def __init__(self, stateSpaceShape, numActions, discountRate, k=math.e):
        self.discountRate = discountRate
        self.numActions = numActions
        self.stateSpaceShape = stateSpaceShape
        #print("state space:",self.stateSpaceShape)
        self.k = k
        self.visitCount = {}
        self.QValues = {}

        for i in range(stateSpaceShape[0]):
            for j in range(stateSpaceShape[1]):
                tempSpace = (i, j)
                self.visitCount[tempSpace] = np.zeros(numActions)
                self.QValues[tempSpace] = np.zeros(numActions)    
        #4 Dimensional
        #for i in range(stateSpaceShape[0]):
        #    for j in range(stateSpaceShape[1]):
        #        for k in range(stateSpaceShape[2]):
        #            for l in range(stateSpaceShape[3]):
        #                tempSpace = (i, j, k, l)
        #                self.visitCount[tempSpace] = np.zeros(numActions)
        #                self.QValues[tempSpace] = np.zeros(numActions)

        #self.model = QLearningNeuralNetwork()
        #self.lossFunction = torch.nn.MSELoss(reduction='sum')
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        pass

    def GetAction_Model(self, currentState, learningMode, randomActionRate=0.00, actionProbabilityBase=0.0):
        #print("Current State:", currentState)
        #stateTensor = torch.Tensor(currentState).cuda()
        if  learningMode == True and np.random.random() < randomActionRate:
            action = np.random.randint(0, self.numActions)
        else:
            QValues = self.GetQValues(currentState)
            actionTensor = torch.argmax(QValues)
            action = actionTensor.item()
        return action

    def GetAction(self, currentStateList, printValues, learningMode, randomActionRate=0.00, actionProbabilityBase=0.0):
        currentState = tuple(currentStateList)
        #print("current state:",currentState)
        if  learningMode == True and np.random.random() < randomActionRate:
            action = np.random.randint(0, self.numActions)
        else:
            if printValues:
                print("Q Values for ",currentState,": ",self.QValues[currentState])
            actionProbs = self.CalculateStateProbabilities(currentState, actionProbabilityBase, printValues) 

            #If all values in the array are equal, pick index randomly
            if np.unique(actionProbs).size == 1:
                action = np.random.randint(0, self.numActions)
            else:
                action = np.argmax(actionProbs)
            if printValues:
                print("Action Probs: ",action)


            #if action == 1:
                #print("Action ", action)
        self.visitCount[currentState][action] += 1
        return action 

    def GetQValues(self, state):
        qValues = []
        for i in range(self.numActions):
            currQ = 0
            qValues.append(currQ)
        return qValues

    def CalculateStateProbabilities(self, state, actionProbabilityBase, printValues):
        currActions = self.QValues[state]
        #currTotal = np.sum(np.power(actionProbabilityBase, currActions))
        return np.divide(np.power(actionProbabilityBase, currActions), np.sum(np.power(actionProbabilityBase, currActions)))
    """
    def CalculateStateProbabilities(self, state, actionProbabilityBase, printValues):
        currActions = self.QValues[state]
        #print("Current Actions:",currActions)
        currTotal = np.sum(np.power(actionProbabilityBase, currActions))  
        if currTotal == 0:
            currTotal = 1  

        actionTotals = []

        for i in range(self.numActions):
            currActionTotal = np.power(actionProbabilityBase, currActions[i])
            if currActionTotal == 0:
                actionTotals.append(1)
            else:
                actionTotals.append(currActionTotal)
        if printValues:
            print("CurrActions:",actionTotals," total:",currTotal)
        currProbabilities = np.divide(actionTotals, currTotal)
        if printValues:
            print("Probabilities:",currProbabilities)
        return currProbabilities
          
    def CalculateStateProbabilities(self, state, actionProbabilityBase, printValues):
        currActions = self.QValues[state]
        actionSum = 0
        actionTotals = []

        for i in range(self.numActions):
            if printValues:
                print("prob base:", actionProbabilityBase," currAction:", currActions[i])
            curr_Pais = actionProbabilityBase ** currActions[i]
            actionSum += curr_Pais
            if curr_Pais == 0:
                actionTotals.append(1)
            else:
                actionTotals.append(curr_Pais)

        currProbabilities = []

        if printValues:
            print("Action sum:",actionSum)

        for i in range(self.numActions):
            currProbabilities.append(actionTotals[i] / actionSum) 
        if printValues:
            print("Probabilities:",currProbabilities)
        return currProbabilities
    """
    def alpha(self, state, action, learningRateScale):
        return np.divide(1, np.add(1, np.multiply(learningRateScale, self.visitCount[state][action])))

    def ObserveAction(self, oldStateList, action, newStateList, reward, learningRateScale, printValues=False):
        oldState = tuple(oldStateList)
        newState = tuple(newStateList)

        self.visitCount[oldState][action] += 1


        alphaValue = self.alpha(oldState, action, learningRateScale)
        newStateValues = self.QValues[newState]
        oldStateValues = self.QValues[oldState]
        #print("NewState Values:", oldStateValues)
        #print("(1-",alphaValue,")*",self.QValues[oldState][action]," + ",alphaValue,"*(",reward,"+",self.discountRate,"*",newStateValues[np.argmax(newStateValues)],")")
        oldStateValues[action] = np.add(np.multiply(np.subtract(1, alphaValue), oldStateValues[action]), np.multiply(alphaValue, np.add(reward, np.multiply(self.discountRate, newStateValues[np.argmax(newStateValues)]))))
        #print("NewState Values Now:", self.QValues[oldStateValues])

class QLearningUnittest():
    def test_2(self):
        qlearner = QLearning([2,2], 3, 0.98)
        currentState = (0,0)
        print(qlearner.QValues)
        qlearner.QValues[currentState] = [-45.53558175, -45.31415765, -45.17032977]
        print("Q Values for ",currentState,": ",qlearner.QValues[currentState])

        actionProbs = qlearner.CalculateStateProbabilities(currentState, 1.8, True)
        input("Troubleshoot")

    def test_1(self):
        qlearner = QLearning([2,2], 2, 0.9)

        print("action 1")
        qlearner.ObserveAction([0,0], 1, [0,1], 1, learningRateScale = 1.0)
        print(qlearner.QValues)
        print("visit")
        print(qlearner.visitCount)

        print("action 2")
        qlearner.ObserveAction([0,0], 1, [0,1], 1, learningRateScale = 1.0)
        print(qlearner.QValues)
        print("visit")
        print(qlearner.visitCount)     

    #def GetAction_Model(self, currentState, learningMode, randomActionRate=0.00, actionProbabilityBase=0.0):
        #if  learningMode == True and np.random.random() < randomActionRate:
        #    action = np.random.randint(0, self.numActions)
        #else:   
        #    QValues = self.model(currentState)
        #    actionTensor = torch.argmax(QValues)
        #    action = actionTensor.item()
    #    return 0
"""
class QLearningNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(QLearningNeuralNetwork, self).__init__()
        
        torch.cuda.set_device(0)

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(4, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(20, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            ).cuda()

    def forward(self, x):
        #print("input shape ", len(x))
        x = np.reshape(x, [-1, len(x)])
        out = torch.Tensor(x).cuda()

        #print("Input shape reshaped:",out.shape)

        out = func.relu(self.fullyConnectedOne(out))
        #print("Input shape fully 1:",out.shape)

        out = func.relu(self.fullyConnectedTwo(out))
        #print("Input shape fully 2:",out.shape)

        out = self.outputLayer(out)
        #print("Input shape output:",out.shape)

        return out
"""