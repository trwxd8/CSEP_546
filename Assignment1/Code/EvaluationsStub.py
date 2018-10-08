

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == 1 and yPredicted[i] == 1):
            correct.append(1)
        else:
            correct.append(0)

    return EnsureDenominatorNonZero(sum(correct), sum(yPredicted))

def Recall(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    
    true_pos = false_neg = 0
    for i in range(len(y)):
        if(y[i] == 1):
            if(yPredicted[i] == 1):
                true_pos += 1
            else:
                false_neg += 1

    return EnsureDenominatorNonZero(true_pos, (true_pos + false_neg))   
	
def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    
    false_pos = true_neg = 0
    for i in range(len(y)):
        if(y[i] == 0):
            if(yPredicted[i] == 1):
                false_pos += 1
            else:
                true_neg += 1

    return EnsureDenominatorNonZero(false_pos, (false_pos + true_neg))   


def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    false_neg = true_pos = 0
    for i in range(len(y)):
        if(y[i] == 1):
            if(yPredicted[i] == 0):
                false_neg += 1
            else:
                true_pos += 1

    return EnsureDenominatorNonZero(false_neg, (true_pos + false_neg))   


def ConfusionMatrix(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    true_pos = false_pos = true_neg = false_neg = 0
    for i in range(len(y)):
        if(y[i] == 1):
            if(yPredicted[i] == 1):
                true_pos += 1
            else:
                false_neg += 1
        elif(y[i] == 0):
            if(yPredicted[i] == 0):
                true_neg += 1
            else:
                false_pos += 1

    print(" |----------------------------------|\n",
          "|             |     Prediction     |\n",
          "|             ---------------------|\n",
          "|             |    1    |    0     |\n",
          "|----------------------------------|\n",
          "|         | 1 |  %5d  |  %5d   |\n" % (true_pos, false_neg),
          "| Actual  |---|--------------------|\n",
          "|         | 0 |  %5d  |  %5d   |\n" % (false_pos, true_neg),
          "|----------------------------------|\n",)

def EnsureDenominatorNonZero(numerator, denominator):
    if(denominator == 0):
        return numerator
    else:
        return numerator/denominator 


def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    
