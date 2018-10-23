def DefineDataBounds(xTrainRaw, yTrainRaw, k, i):
    cnt = len(xTrainRaw)
    subset_size = cnt / k
    validateStartIdx = subset_size * (i)
    validateStopIdx = subset_size * (i + 1)

    xTrainOnRaw = []
    xValidateOnRaw = []
    yTrainOnRaw = []
    yValidateOnRaw = []

    #USE Bracket notation validate = xTrainRaw[validateStartIdx:validateStopIdx]
    for idx in range(cnt):
        if idx >= validateStartIdx and idx < validateStopIdx:
            xValidateOnRaw.append(xTrainRaw[idx])
            yValidateOnRaw.append(yTrainRaw[idx])
        else:
            xTrainOnRaw.append(xTrainRaw[idx])
            yTrainOnRaw.append(yTrainRaw[idx])

    return (xTrainOnRaw, yTrainOnRaw, xValidateOnRaw, yValidateOnRaw)
