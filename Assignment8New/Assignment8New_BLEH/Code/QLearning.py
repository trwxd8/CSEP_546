import numpy as np
import torch
import torch.nn.functional as func
import math

class QLearning():
    def __init__(self, stateSpaceShape, numActions, discountRate, k=math.e):
        self.discountRate = discountRate
        self.numActions = numActions
        self.stateSpaceShape = stateSpaceShape
        print("state space:",self.stateSpaceShape)
        self.k = k
        input("Test")
        self.visitCount = {}
        self.QValues = {}

        for i in range(stateSpaceShape[0]):
            for j in range(stateSpaceShape[1]):
                for k in range(stateSpaceShape[2]):
                    for l in range(stateSpaceShape[3]):
                        tempSpace = [i, j, k, l]
                        self.visitCount[tempSpace] = np.zeros(numActions)
                        self.QValues[tempSpace] = np.zeros(numActions)

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

    def GetAction(self, currentState, learningMode, randomActionRate=0.00, actionProbabilityBase=0.0):
        print("current state:")
        if  learningMode == True and np.random.random() < randomActionRate:
            action = np.random.randint(0, self.numActions)
        else:
             actionProbs = self.CalculateStateProbabilities(currentState, actionProbabilityBase) 
             action = 0
        print("Action ", action)
        return action 

    def GetQValues(self, state):
        qValues = []
        for i in range(self.numActions):
            currQ = 0
            qValues.append(currQ)
        return qValues

    def CalculateStateProbabilities(self, state, actionProbabilityBase):
        currActions = self.QValues[state]
        currTotal = np.sum(np.power(self.k, currActions))    
        
        return np.divide(np.power(self.k, currActions), currTotal)
            
    def alpha(self, state, action):
        return np.divide(1, np.add(1, self.visitCount[(state, action)]))

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale):
        alphaValue = self.alpha(oldState, action)
        newStateValues = self.QValues[newState]
        newStateValues[action] = np.add(np.multiply(np.subtract(1, alphaValue, self.QValues[state][action]), np.multiply(alphaValue, np.add(reward, np.multiply(self.discountRate, newStateValues[np.argmax(newStateValues)]))))

"""    def GetAction_Model(self, currentState, learningMode, randomActionRate=0.00, actionProbabilityBase=0.0):
        if  learningMode == True and np.random.random() < randomActionRate:
            action = np.random.randint(0, self.numActions)
        else:
            
            QValues = self.model(currentState)
            actionTensor = torch.argmax(QValues)
            action = actionTensor.item()
        return action
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
