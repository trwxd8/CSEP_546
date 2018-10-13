import math
import numpy as np

class LogisticRegressionModel(object):
    """A model that calculates the logsitics regression analysis."""

    def __init__(self):
        pass

    def fit(self, x, y, iterations, step):
        #self.weight0 = .05
        #self.weights = [.05, -.05, .05, -.05, .05]

        #Initialize weights to arbitrary values
        self.weight0 = .5
        self.weights = [.75, .75, .75, .25, .25]

        #Go through iteration number of steps for optimizing weights
        for curr_iter in range(iterations+1):
            yPredictions = self.calculateSigmoids(x)

            #calculate training loss with respect to weight
            cnt = len(yPredictions)
            w_cnt = len(self.weights)

            #Adjust weight0
            summed_loss = sum([(yPredictions[j] - y[j]) for j in range(cnt)])
            trainset_loss = summed_loss/cnt
            self.weight0 -= (step * trainset_loss)

            #Adjust weight for each
            for i in range(w_cnt):
                summed_loss = sum([(yPredictions[j] - y[j])*x[j][i] for j in range(cnt)])
                trainset_loss = summed_loss/cnt
                self.weights[i] -= (step * trainset_loss) 
    
            #Include printing of test loss
            #if(curr_iter % 1000 == 0):
            #    print(curr_iter,"training loss:",self.loss(xTest, yTest))
            #if(curr_iter % 1000 == 0 and curr_iter != 0):
            #    print(curr_iter," training set loss:", self.loss(x, y))
    

        return 0

    def predict(self, x):
        predictions = []
        y_Predictions = self.calculateSigmoids(x)

        for yPredicted in y_Predictions:
            if yPredicted > .5:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions

    def calculateSigmoids(self, x):
        yPredicted = []

        for example in x:
            z = self.weight0 + np.dot(example, self.weights)
            sigmoid = 1.0 / (1.0 + math.exp(-1*z))
            yPredicted.append(sigmoid)
        return yPredicted

    def loss(self, x, y):
        losses = []

        yPredicted = self.calculateSigmoids(x)

        for i in range(len(y)):
            losses.append((-1*y[i] * math.log(yPredicted[i])) - ((1 - y[i]) * (math.log(1.0-yPredicted[i]))))

        return sum(losses)