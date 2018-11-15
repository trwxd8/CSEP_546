import math

def Get95LowerAndUpperBounds(Accuracy, n):
    lowerBound = Accuracy - 1.96 * math.sqrt((Accuracy * (1 - Accuracy) / n))
    upperBound = Accuracy + 1.96 * math.sqrt((Accuracy * (1 - Accuracy) / n))
    return ( lowerBound, upperBound )