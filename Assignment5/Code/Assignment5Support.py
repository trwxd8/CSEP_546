import os
import numpy as np
import random


def LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True, shuffle=True):
    xRaw = []
    yRaw = []
    
    if includeLeftEye:
        closedEyeDir = os.path.join(kDataPath, "closedLeftEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openLeftEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if includeRightEye:
        closedEyeDir = os.path.join(kDataPath, "closedRightEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openRightEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if shuffle:
        random.seed(1000)

        index = [i for i in range(len(xRaw))]
        random.shuffle(index)

        xOrig = xRaw
        xRaw = []

        yOrig = yRaw
        yRaw = []

        for i in index:
            xRaw.append(xOrig[i])
            yRaw.append(yOrig[i])

    return (xRaw, yRaw)


from PIL import Image

def Convolution3x3(image, filter):
    # check that the filter is formated correctly
    if not (len(filter) == 3 and len(filter[0]) == 3 and len(filter[1]) == 3 and len(filter[2]) == 3):
        raise UserWarning("Filter is not formatted correctly, should be [[x,x,x], [x,x,x], [x,x,x]]")

    xSize = image.size[0]
    ySize = image.size[1]
    pixels = image.load()

    answer = []
    for x in range(xSize):
        answer.append([ 0 for y in range(ySize) ])

    # skip the edges
    for x in range(1, xSize - 1):
        for y in range(1, ySize - 1):
            value = 0

            for filterX in range(len(filter)):
                for filterY in range(len(filter)):
                    imageX = x + (filterX - 1)
                    imageY = y + (filterY - 1)

                    value += pixels[imageX, imageY] * filter[filterX][filterY]

            answer[x][y] = value

    return answer

def Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeXGradients=True, includeYGradients=True, includeYHistogram=True, includeXHistogram=True, includeRawPixels=False, includeIntensities=False):
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for sample in xTrainRaw:
        features = []

        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            yEdges = Convolution3x3(image, [[1, 0, -1],[2,0,-2],[1,0,-1]])
            sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            count = sum([len(row) for row in yEdges])

            features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            count = sum([len(row[8:16]) for row in yEdges])

            features.append(sumGradient / count)

        if includeYGradients or includeYHistogram:
            yGradientImage =  np.array(Convolution3x3(image, [[1, 0, -1],[2,0,-2],[1,0,-1]]))

            if includeYGradients:
                gradientFeatures = CalculateGradientFeatures(yGradientImage)
                for currFeature in gradientFeatures:
                    features.append(currFeature)

            if includeYHistogram:
                yHistogram = CalculateHistogramFeatures(yGradientImage)
                for histCnt in yHistogram:
                    features.append(histCnt)

        if includeXGradients or includeXHistogram:
            xGradientImage = np.array(Convolution3x3(image, [[1, 2, 1],[0,0,0],[-1,-2,-1]]))

            if  includeXGradients:
                gradientFeatures = CalculateGradientFeatures(xGradientImage)
                for currFeature in gradientFeatures:
                    features.append(currFeature)

            if includeXHistogram:
                xHistogram = CalculateHistogramFeatures(xGradientImage)
                for histCnt in xHistogram:
                    features.append(histCnt)

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])


        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for sample in xTestRaw:
        features = []
        
        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            yEdges = Convolution3x3(image, [[1, 0, -1],[2,0,-2],[1,0,-1]])
            sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            count = sum([len(row) for row in yEdges])

            features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            count = sum([len(row[8:16]) for row in yEdges])

            features.append(sumGradient / count)

        if includeYGradients or includeYHistogram:
            yGradientImage =  np.array(Convolution3x3(image, [[1, 0, -1],[2,0,-2],[1,0,-1]]))

            if includeYGradients:
                gradientFeatures = CalculateGradientFeatures(yGradientImage)
                for currFeature in gradientFeatures:
                    features.append(currFeature)

            if includeYHistogram:
                yHistogram = CalculateHistogramFeatures(yGradientImage)
                for histCnt in yHistogram:
                    features.append(histCnt)

        if includeXGradients or includeXHistogram:
            xGradientImage = np.array(Convolution3x3(image, [[1, 2, 1],[0,0,0],[-1,-2,-1]]))

            if  includeXGradients:
                gradientFeatures = CalculateGradientFeatures(xGradientImage)
                for currFeature in gradientFeatures:
                    features.append(currFeature)

            if includeXHistogram:
                xHistogram = CalculateHistogramFeatures(xGradientImage)
                for histCnt in xHistogram:
                    features.append(histCnt)

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])

        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        xTest.append(features)

    return (xTrain, xTest)

def CalculateGradientFeatures(gradientImage):
    currFeatures = []
    gridCnt = 3

    xGridSize = int(gradientImage.shape[0] / gridCnt)
    yGridSize = int(gradientImage.shape[1] / gridCnt)

    for yGridIdx in range(gridCnt):
        yStartIdx = yGridIdx * yGridSize
        for xGridIdx in range(gridCnt):
            xStartIdx = xGridIdx * xGridSize
            currMin = 256
            currMax = -1
            currTotal = 0
            for yIdx in range(yGridSize):
                for xIdx in range(xGridSize):
                    currY = yStartIdx + yIdx
                    currX = xStartIdx + xIdx
                    value = abs(gradientImage[currY][currX])
                    if value < currMin:
                        currMin = value
                    if value > currMax:
                        currMax = value
                    currTotal += value
            currAvg = currTotal / (xGridSize * yGridSize)

            currFeatures.append(currMin)
            currFeatures.append(currMax)
            currFeatures.append(currAvg)
    return currFeatures

def CalculateHistogramFeatures(gradientImage):
    xImgSize = int(gradientImage.shape[0])
    yImgSize = int(gradientImage.shape[1]) 

    histogram = np.zeros(5)
    pixelCnt = xImgSize * yImgSize

    for currY in range(yImgSize):
        for currX in range(xImgSize):
            currValue = gradientImage[currX,currY]/255.0

            if currValue >= .8:
                histogram[4] += 1
            elif currValue >= .6:
                histogram[3] += 1
            elif currValue >= .4:
                histogram[2] += 1
            elif currValue >= .2:
                histogram[1] += 1
            else:
                histogram[0] += 1
    return histogram #np.divide(histogram, pixelCnt)

import PIL
from PIL import Image

def VisualizeWeights(weightArray, outputPath):
    size = 12

    # note the extra weight for the bias is where the +1 comes from, just ignore it
    if len(weightArray) != (size*size) + 1:
        raise UserWarning("size of the weight array is %d but it should be %d" % (len(weightArray), (size*size) + 1))

    if not outputPath.endswith(".jpg"):
        raise UserWarning("output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

    image = Image.new("L", (size,size))

    pixels = image.load()

    for x in range(size):
        for y in range(size):
            pixels[x,y] = int(abs(weightArray[(x*size) + y]) * 255)

    image.save(outputPath)