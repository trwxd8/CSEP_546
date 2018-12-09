#THWEID Thomas Weidmaier
# CSEP 546 10/08
# Assignment 1 - Basic Model Evaluation

import math

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

#Calculate true positive and false positive to calculate
#Precision value, true positive / (true positive + false positive)
def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    true_pos = false_pos = 0
    for i in range(len(y)):
        if(yPredicted[i] == 1):
            if(y[i] == 1):
                true_pos += 1
            else:
                false_pos += 1

    return EnsureDenominatorNonZero(true_pos, (true_pos + false_pos))

#Calculate true positive and false positive to calculate
#Recall value, true positive / (true positive + false negative)
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

#Calculate false negative and true positive to calculate
#False negative rate, false negative / (true positive + false negative)
def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    false_neg = true_pos = 0
    for i in range(len(y)):
        if(y[i] == 1):
            if(yPredicted[i] == 0):
                false_neg += 1
            else:
                true_pos += 1

    return EnsureDenominatorNonZero(false_neg, (true_pos + false_neg))   


#Calculate false positive and true negative to calculate
#False positive rate, false positive / (false positive + true negative)
def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    
    false_pos = true_neg = 0
    for i in range(len(y)):
        if(y[i] == 0):
            if(yPredicted[i] == 1):
                false_pos += 1
            else:
                true_neg += 1

    return EnsureDenominatorNonZero(false_pos, (false_pos + true_neg))   


#Print out a confusion matrix containing evaluation metrics
def ConfusionMatrix(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    true_pos = false_pos = true_neg = false_neg = 0
    for i in range(len(y)):
        #Look at actual value 
        if(y[i] == 1):
            #Use predicted to increase count of true positive if matching, or false negative if not
            if(yPredicted[i] == 1):
                true_pos += 1
            else:
                false_neg += 1
        #compare for non-1 values
        elif(y[i] == 0):
            #Use predicted to increase count of true negative if matching, or false positive if not
            if(yPredicted[i] == 0):
                true_neg += 1
            else:
                false_pos += 1

    #Make calls to filled out functions
    curr_accuracy = Accuracy(y, yPredicted)
    curr_precision = Precision(y, yPredicted)
    curr_recall = Recall(y, yPredicted)
    curr_FalsePositiveRate = FalsePositiveRate(y, yPredicted)
    curr_FalseNegativeRate = FalseNegativeRate(y, yPredicted)

    #Print information in table format
    print(" |-----------------------------------------------------------|\n",
           "|             |      Prediction       |                     |\n",
           "|             |-----------------------|                     |\n",
           "|             |     1     |     0     |                     |\n",
           "|-------------|-----------|-----------|---------------------|\n",
           "|         | 1 |  %5d    |  %5d    | Recall  |%.9f|\n" % (true_pos, false_neg, curr_recall),
           "| Actual  |---|-----------|-----------|---------|-----------|\n",
           "|         | 0 |  %5d    |  %5d    |   FPR   |%.9f|\n" % (false_pos, true_neg, curr_FalsePositiveRate),
           "|-------------|-----------|-----------|---------------------|\n",
           "|             | Precision |    FNR    |      Accuracy:      |\n",
           "|             |-----------|-----------|     %.9f     |\n" % (curr_accuracy), 
           "|             |%.9f|%.9f|                     |\n" % (curr_precision, curr_FalseNegativeRate),                   
           "|-----------------------------------------------------------|\n",)


# Function to check that there is no division by 0,
# returning zero if denominator is 0, as discussed in slack
def EnsureDenominatorNonZero(numerator, denominator):
    if(denominator == 0):
        return 0
    else:
        return numerator/denominator 


def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))