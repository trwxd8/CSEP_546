import math

class LogisticRegressionModel(object):
    """A model that calculates the logsitics regression analysis."""

    def __init__(self):
        pass

    def fit(self, x, y, iterations, step):
        self.weights = [.75, .75, .75, .25, .25]
        pass

    def predict(self, x):
        predictions = []
        cnt = 0
        for example in x:
            z = self.weights[0] + sum([ example[i] * self.weights[i] for i in range(len(example)) ])
            yPredicted = self.sigmoid(z)
            print(cnt,":",yPredicted)
            if yPredicted > .5:
                predictions.append(1)
            else:
                predictions.append(0)
            cnt += 1
        
        return predictions

    def sigmoid(self, x):
        return 1 / (1.0 + math.exp(-1*x))

    def loss(self, y, yPredicted):
        losses = []

        for i in range(len(y)):
            print(i,":",yPredicted[i])
            losses[i] = (-1*y[i] * math.log(yPredicted[i])) - ((1 - y[i]) * (math.log(1.0-yPredicted[i])))

        return sum(losses)