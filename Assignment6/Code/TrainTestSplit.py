def TrainTestSplit(xRaw, yRaw, percentTest):
    xLen = len(xRaw)
    yLen = len(yRaw)
    if(xLen != yLen):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(xLen * percentTest)

    if(numTest == 0 or numTest > yLen):
        raise UserWarning("Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = xRaw[:numTest]
    xTrain = xRaw[numTest:]
    yTest = yRaw[:numTest]
    yTrain = yRaw[numTest:]

    return (xTrain, yTrain, xTest, yTest)