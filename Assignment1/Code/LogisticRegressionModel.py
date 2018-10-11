import math

class LogisticRegressionModel(object):
    """A model that calculates the logsitics regression analysis."""

    def __init__(self):
        pass

    def fit(self, x, y, iterations, step):
        #self.weight0 = .05
        #self.weights = [.05, -.05, .05, -.05, .05]

        self.weight0 = .5
        self.weights = [.75, .75, .75, .25, .25]

        for curr_iter in range(iterations):
            yPredictions = self.find_regression(x)

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
                #print(i,": summed loss-",summed_loss," loss-", trainset_loss, " new weight-", self.weights[i])
            #print(curr_iter,":",summed_loss,",",self.weights)
    
    

        return 0

    def predict(self, x):
        predictions = []
        y_Predictions = self.find_regression(x)

        for yPredicted in y_Predictions:
            if yPredicted > .5:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-1*x))

    def find_regression(self, x):
        weighted_values = []
        for example in x:
            z = self.weight0 + sum([ example[i] * self.weights[i] for i in range(len(example)) ])
            weighted_values.append(self.sigmoid(z))
        return weighted_values

    def loss(self, x, y):
        losses = []

        yPredicted = self.find_regression(x)

        for i in range(len(y)):
            losses.append((-1*y[i] * math.log(yPredicted[i])) - ((1 - y[i]) * (math.log(1.0-yPredicted[i]))))

        return sum(losses)