import copy
import math
import numpy as np
import random as rand

class DecisionTreeModel(object):
    """A model that creates a model based off a decision tree."""

    def __init__(self):
        pass

    def fit(self, x, y, minSplits, featureRestriction, seed):
        self.minSplits = minSplits
        np_x = np.asarray(x)
        np_y = np.asarray(y)
        
        if featureRestriction >= 0:
            bitmap = self.restrictFeatures(len(x[0]), featureRestriction, seed)
        else:
            bitmap = np.ones(len(x[0]))
        print(bitmap)

        #Start decision tree model
        self.root = self.growTree(np_x, np_y, bitmap, seed)
        pass

    def predict(self, x, predictionThreshold):
        predictions = []        
        for example in x:
            predictions.append(self.predictMessage(example, self.root, predictionThreshold))
        return predictions
    
    def predictMessage(self, example, node, predictionThreshold):
        #If a leaf has been reached, return the value
        if node.Feature == None:      
            return 0 if node.Value < predictionThreshold else 1

        #Otherwise, descend in the tree, choosing the direction based on the feature value
        elif example[node.Feature] < node.threshold:
            return self.predictMessage(example, node.LeftNode, predictionThreshold)
        else:
            return self.predictMessage(example, node.RightNode, predictionThreshold)


    def growForest(self, x, y, numTrees, minSplit, baggingFlag, featureRestriction, seed):

        forest = []
        for i in range(numTrees):
            
            #create modified sets of X and Y if bagging applied
            if baggingFlag == True:
                (x, y) = self.bagDataset(x, y, seed)

            #Calculate decision tree instance and add to forest
            forest.append(self.fit(x, y, minSplit, featureRestriction, seed))
        pass

    def bagDataset(self, x, y, seed):
        cnt = len(y)
        rand.seed(seed)
        baggedX = []
        baggedY = []

        for i in range(cnt):
            randIdx = rand.randrange(cnt)
            baggedX.append(x[randIdx])
            baggedY.append(y[randIdx])

        return (baggedX, baggedY)

    def restrictFeatures(self, bitmapSize, featureCnt, seed):
        rand.seed(seed)
        bitmap = np.zeros(bitmapSize)

        i = 0
        while i < featureCnt:
            #Pick a random feature and if it hasn't been enabled, do so
            idx = rand.randrange(bitmapSize)
            
            if bitmap[idx] != 1:
                bitmap[idx] = 1
                i += 1
        return bitmap


    def growTree(self, x, y, bitmap, level):
        #print("level:",level)
        value, counts = np.unique(y, return_counts=True)
        instanceDict = dict(zip(value, counts))
        yLen = len(y)

        #If below minimum threshold/no more features to split on/all values the same,
        # create leaf with probability that result is 1 as value
        if yLen < self.minSplits or np.isin(1, bitmap) == False or np.isin(0, y) == False or np.isin(1, y) == False:
            if np.isin(1, y) == False:
                oneCnt = 0
            else:
                oneCnt = instanceDict[1]
            return DecisionTreeNode(None, None, None, (oneCnt + 1) / (yLen + 2), None)

        #Pick feature to split on 
        xIdx = self.PickBestSplit(x, y, bitmap)

        #If no information gain present, create leaf
        if xIdx == -1:
            if np.isin(1, y) == False:
                oneCnt = 0
            else:
                oneCnt = instanceDict[1]
            return DecisionTreeNode(None, None, None, (oneCnt + 1) / (yLen + 2), None)

        if xIdx != 0:
            bitmap[xIdx] = 0

        #Split datasets on value for feature
        (threshold, xSplitLeft, ySplitLeft, xSplitRight, ySplitRight) = self.SplitByFeature(x, y, xIdx)
        leftBitmap = np.asarray(copy.deepcopy(bitmap))
        rightBitmap = np.asarray(copy.deepcopy(bitmap))

        return DecisionTreeNode(self.growTree(xSplitLeft, ySplitLeft, leftBitmap, level + 1), self.growTree(xSplitRight, ySplitRight, rightBitmap, level + 1), xIdx, None, threshold)

    def PickBestSplit(self, x, y, bitmap):
        infoGains = []
        cntFeatures = len(x[0])

        #Calculate information gain for each feature
        for idx in range(cntFeatures):
            #If feature has already been used, don't try again
            if bitmap[idx] == 0:
                infoGains.append(-1)
            else:
                #print("finding gain for ",idx,"/",cntFeatures)
                infoGains.append(self.InformationGains(x, y, idx))
        
        #Return index with most information gain, or -1 if no information is gained
        maxIdx = np.argmax(infoGains)
        if infoGains[maxIdx] < 0:
            return -1
        else:
            return maxIdx 

    def InformationGains(self, x, y, idx):
        entropyY = self.Entropy(y)
        lossXidx = self.Loss(x, y, idx)
        return entropyY - lossXidx

    def Entropy(self, y):
        value, counts = np.unique(y, return_counts=True)
        instanceDict = dict(zip(value, counts))
        sumEntropy = 0
        cnt = np.size(y)

        for currKey in instanceDict:
            currProb = (instanceDict[currKey]) / (cnt)
            sumEntropy += currProb * math.log2(currProb)

        return -sumEntropy

    def Loss(self, x, y, idx):
        #print("Loss at ",idx)
        xTransposed = np.transpose(x)
        xIdxDataset = xTransposed[idx]
        value, counts = np.unique(xIdxDataset, return_counts=True)
        instanceDict = dict(zip(value, counts))
        xLen = xIdxDataset.shape[0]
        lossSum = 0

        #Get set of y values for each x feature value
        for currKey in instanceDict:
            currY = []
            for i in range(xLen):
                if xIdxDataset[i] == currKey:
                    currY.append(y[i])

            #Calculate the entropy for current value
            lossSum += self.Entropy(currY) * instanceDict[currKey]
        return lossSum / xLen


    #Function to extract subset of data based on value for feature at idx
    def SplitByFeature(self, x, y, idx):
        xSplitLeft = []
        ySplitLeft = []
        xSplitRight = []
        ySplitRight = []

        xTransposed = np.transpose(x)
        xIdxDataset = xTransposed[idx]

        min = np.amin(xIdxDataset)
        max = np.amax(xIdxDataset)
        threshold = min + (max - min) / 2
        #print("min:",min," max:",max," threshold:",)

        cnt = len(y)
        for curr in range(cnt):
            if x[curr][idx] < threshold:
                xSplitLeft.append(x[curr])
                ySplitLeft.append(y[curr])
            else:
                xSplitRight.append(x[curr])
                ySplitRight.append(y[curr])
        return (threshold, np.asarray(xSplitLeft), np.asarray(ySplitLeft), np.asarray(xSplitRight), np.asarray(ySplitRight))

    def PrintTree(self, threshold):
        self.printNode(self.root, "", threshold)

    def printNode(self, node, spacing, predictionThreshold):
        if node == None:
            return
        if node.Feature == None:
            nodeValue = 0 if node.Value < predictionThreshold else 1
            print(spacing + str(nodeValue))
            return
        
        print(spacing + "Feature",node.Feature+1,":")           
        newSpacing = spacing + "     "
        print(spacing,"  <", node.threshold)
        self.printNode(node.LeftNode, newSpacing, predictionThreshold)
        print(spacing,"  >=", node.threshold)
        self.printNode(node.RightNode, newSpacing, predictionThreshold)

class DecisionTreeNode(object):
    
    def __init__(self, leftNode, RightNode, idx, value, threshold):
        self.LeftNode = leftNode
        self.RightNode = RightNode
        self.Feature = idx
        self.Value = value
        self.threshold = threshold

class DecisionTreeUnitTest(object):
    def __init__(self):
        self.y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.x = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 1, 0, 1],
                           [1, 1, 0, 1],
                           [1, 1, 0, 2],
                           [2, 1, 1, 2],
                           [2, 1, 1, 2],
                           [2, 1, 1, 2],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 1, 1],
                           [2, 0, 1, 2],
                           [2, 1, 1, 2]])

    def ExecuteTest(self):
        #print("x:", self.x)
        #print("y:", self.y)
        decisionTree = DecisionTreeModel()
        decisionTree.PickBestSplit(self.x, self.y, [1, 1, 1, 1])

