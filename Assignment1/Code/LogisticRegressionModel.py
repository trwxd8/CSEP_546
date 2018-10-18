import math
import numpy as np

class LogisticRegressionModel(object):
    """A model that calculates the logsitics regression analysis."""

    def __init__(self):
        pass

    def fit(self, x, y, iterations, step):
        
		#Create random weights bounded between -.05 and .05
        w_cnt = len(x[0])
        self.weights = np.multiply(np.subtract(np.random.ranf(w_cnt), .5), .1)

        #Go through iteration number of steps for optimizing weights
        for curr_iter in range(iterations):
            yPredictions = self.calculateSigmoids(x)

            #calculate training loss with respect to weight
            cnt = len(yPredictions)
			#Transpose the x values so dot multiply can be used
            t_x = np.matrix.transpose(x)
			
            #for i in range(w_cnt):
            #    summed_loss = sum([(yPredictions[j]-y[j])*x[j][i] for j in range(cnt)])
            #    trainset_loss = summed_loss/cnt
            #    self.weights[i] -= (step * trainset_loss) 
            self.weights = np.subtract(self.weights, np.multiply(step, np.divide(np.dot(t_x, np.subtract(yPredictions, y)), cnt)))

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
        #yPredicted = []
        #zList = np.dot(x, self.weights)
		
        #for z in zList:
        #    sigmoid = 1.0 / (1.0 + math.exp(-1*z))
        #    yPredicted.append(sigmoid)
        #return yPredicted
        return np.divide(1.0, np.add(1.0, np.exp(np.multiply(-1.0, np.dot(x, self.weights)))))
		
    def loss(self, x, y):
        yPredicted = self.calculateSigmoids(x)
        #losses = []
        #for i in range(len(y)):
        #    losses.append((-1*y[i] * math.log(yPredicted[i])) - ((1 - y[i]) * (math.log(1.0-yPredicted[i]))))
        #return sum(losses)
        return np.sum(np.subtract(np.multiply(-y, np.log(yPredicted)), np.multiply(np.subtract(1, y), np.log(np.subtract(1.0,yPredicted)))))