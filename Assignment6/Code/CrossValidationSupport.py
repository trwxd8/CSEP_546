def DefineDataBounds(xTrain, yTrain, k, i):
    cnt = len(xTrain)
    subset_size = cnt / k
    validateStartIdx = subset_size * (i)
    validateStopIdx = subset_size * (i + 1)

    xTrainOn = []
    xValidateOn = []
    yTrainOn = []
    yValidateOn = []

    #USE Bracket notation validate = xTrainRaw[validateStartIdx:validateStopIdx]
    for idx in range(cnt):
        if idx >= validateStartIdx and idx < validateStopIdx:
            xValidateOn.append(xTrain[idx])
            yValidateOn.append(yTrain[idx])
        else:
            xTrainOn.append(xTrain[idx])
            yTrainOn.append(yTrain[idx])

    return (xTrainOn, yTrainOn, xValidateOn, yValidateOn)
