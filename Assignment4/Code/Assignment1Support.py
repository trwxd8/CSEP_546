import collections
import BagOfWordsModel
import numpy as np

def LoadRawData(path):
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 5574

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:])
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)

def TrainTestSplit(x, y, percentTest = .25):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)

def FeaturizeHandcrafted(xTrainRaw, xTestRaw):
    words = ['call', 'to', 'your']

    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        # Have a feature for longer texts
        if(len(x)>40):
            features.append(1)
        else:
            features.append(0)

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        fullWords = x.split()
        for word in words:
            if word in fullWords:
                features.append(1)
            else:
                features.append(0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for x in xTestRaw:
        features = []
        
        # Have a feature for longer texts
        if(len(x)>40):
            features.append(1)
        else:
            features.append(0)

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        fullWords = x.split()
        for word in words:
            if word in fullWords:
                features.append(1)
            else:
                features.append(0)

        xTest.append(features)

    return (xTrain, xTest)

def FeaturizeContinuousLength(xTrainRaw, xTestRaw):
    words = ['call', 'to', 'your']

    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        # Have a feature for longer texts
        features.append(len(x))

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        fullWords = x.split()
        for word in words:
            if word in fullWords:
                features.append(1)
            else:
                features.append(0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for x in xTestRaw:
        features = []
        
        # Have a feature for longer texts
        features.append(len(x))

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        fullWords = x.split()
        for word in words:
            if word in fullWords:
                features.append(1)
            else:
                features.append(0)

        xTest.append(features)

    return (xTrain, xTest)

def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])

def Featurize(xTrainRaw, yTrainRaw, xTestRaw, numFrequentWords, numMutualInformationWords, includeHandCraftedFeatures):    
    bagModel = BagOfWordsModel.BagOfWordsModel()
    bagModel.fillVocabulary(xTrainRaw)

    usedWords = []
    xTrain = []
    xTest = []

    print("Handcrafted Features:",includeHandCraftedFeatures," NumFrequentWords:",numFrequentWords," NumMutualInformation:",numMutualInformationWords)

    if includeHandCraftedFeatures == True:
        (xTrainHC, xTestHC) = FeaturizeHandcrafted(xTrainRaw, xTestRaw)      
        xTrain = combineDatasets(xTrain, xTrainHC)
        xTest = combineDatasets(xTest, xTestHC)

        usedWords.append('call')
        usedWords.append('to')
        usedWords.append('your')


    if numFrequentWords > 0:
        results = bagModel.FrequencyFeatureSelection(xTrainRaw, numFrequentWords)
        freqWords = []
        for i in range(numFrequentWords):
            freqWords.append(results[i][0])

        xTrainFreq, xTestFreq = bagModel.FeaturizeByWords(xTrainRaw, xTestRaw, freqWords)
        xTrain = combineDatasets(xTrain, xTrainFreq)
        xTest = combineDatasets(xTest, xTestFreq)


    if numMutualInformationWords > 0:
        results = bagModel.MutualInformationFeatureSelection(xTrainRaw, yTrainRaw, numMutualInformationWords)
        miWords = []
        for i in range(numMutualInformationWords):
            miWords.append(results[i][0])

        xTrainMI, xTestMI = bagModel.FeaturizeByWords(xTrainRaw, xTestRaw, miWords)
        xTrain = combineDatasets(xTrain, xTrainMI)
        xTest = combineDatasets(xTest, xTestMI) 

    return (xTrain, xTest)

def combineDatasets(datasetOriginal, datasetAdditional):
    if datasetOriginal == []:
        return datasetAdditional
    else:
        cnt = len(datasetAdditional)
        for i in range(cnt):
            datasetOriginal[i] += datasetAdditional[i]
        return datasetOriginal