import copy
import math
import numpy as np

class DecisionTreeModel(object):
    """A model that creates a model based off a decision tree."""

    def __init__(self):
        pass

    def fit(self, x, y, minSplits):
        self.minSplits = minSplits
        np_x = np.asarray(x)
        np_y = np.asarray(y)

        cntFeatures = len(x[0])
        bitmap = []
        #Calculate information gain for each feature
        for idx in range(cntFeatures):
            bitmap.append(1)

        self.root = self.growTree(np_x, np_y, bitmap, 0)
        pass

    def predict(self, x):
        predictions = []        
        for example in x:
            predictions.append(self.predictMessage(example, self.root))
        return predictions
    
    def predictMessage(self, example, node):
        if node.Feature == None:
            return node.Value
        elif example[node.Feature] < 0.5:
            return self.predictMessage(example, node.LeftNode)
        else:
            return self.predictMessage(example, node.RightNode)

    def growTree(self, x, y, bitmap, level):
        cnt = len(y)

        #print("curr split at level ",level,": x-",x.shape," y-",y.shape)
        #print(bitmap)

        #If below minimum threshold, return value that occurs most 
        if cnt < self.minSplits or (np.isin(1, bitmap) == False):
            value, counts = np.unique(y, return_counts=True)
            instanceDict = dict(zip(value, counts))
            #print(instanceDict)

            if np.isin(0, y) == False:
                #print("Only 1s left")
                zeroCnt = -1
            else:
                zeroCnt = instanceDict[0]
            if np.isin(1, y) == False:
                #print("Only 0s left")
                oneCnt = -1
            else:
                oneCnt = instanceDict[1]
            #print("0:",zeroCnt," 1:",oneCnt)
            #input("Enter input:")
            return DecisionTreeNode(None, None, None, 0 if zeroCnt > oneCnt else 1)
        
        #If all values are the same, return node of value
        elif np.isin(1, y) == False:
            #print("Only 0s left")
            #input("Enter input:")
            return DecisionTreeNode(None, None, None, 0)
        elif np.isin(0, y) == False:
            #print("Only 1s left")
            #input("Enter input:")
            return DecisionTreeNode(None, None, None, 1)

        #Pick feature to split on 
        else:
            #print("Bitmap of features currently ", bitmap)
            xIdx = self.PickBestSplit(x, y, bitmap)
            #print("Best index is ",xIdx)
            if xIdx == -1:
                #print("All information gains are negative...")
                return None
            bitmap[xIdx] = 0

            #Split datasets on value for feature
            (xSplit0, ySplit0) = self.SplitByFeature(x, y, xIdx, 0)
            (xSplit1, ySplit1) = self.SplitByFeature(x, y, xIdx, 1)
            #print("0 split: x-",xSplit0.shape," y-",ySplit0.shape)
            #print("1 split: x-",xSplit1.shape," y-",ySplit1.shape)           
            #input("Enter input:")

        leftBitmap = copy.deepcopy(bitmap)
        rightBitmap = copy.deepcopy(bitmap)

        return DecisionTreeNode(self.growTree(xSplit0, ySplit0, leftBitmap, level + 1), self.growTree(xSplit1, ySplit1, rightBitmap, level + 1), xIdx, -1)

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
        print(infoGains)
        maxIdx = np.argmax(infoGains)
        if infoGains[maxIdx] < 0:
            return -1
        else:
            return maxIdx 

    def InformationGains(self, x, y, idx):
        entropyY = self.Entropy(y)
        #print("y entropy:", entropyY)
        lossXidx = self.Loss(x, y, idx)
        #print("x",idx," loss:",lossXidx)
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
        yClassifications = {}
        xTransposed = np.transpose(x)
        xIdxDataset = xTransposed[idx]
        value, counts = np.unique(xIdxDataset, return_counts=True)
        instanceDict = dict(zip(value, counts))
        xLen = xIdxDataset.shape[0]

        #Get set of x values for 
        for currKey in instanceDict:
            currY = []
            for i in range(xLen):
                if xIdxDataset[i] == currKey:
                    currY.append(y[i])
            yClassifications[currKey] = currY

        lossSum = 0
        #Add loss of 
        for currKey in instanceDict:
            currLoss = self.Entropy(yClassifications[currKey]) * instanceDict[currKey]
            #print("Loss for ",currKey, "is ", currLoss)
            lossSum += currLoss
        return lossSum / xLen


    #Function to extract subset of data based on value for feature at idx
    def SplitByFeature(self, x, y, idx, value):
        xSplit = []
        ySplit = []

        cnt = len(x)
        for curr in range(cnt):
            if x[curr][idx] == value:
                xSplit.append(x[curr])
                ySplit.append(y[curr])
        return (np.asarray(xSplit), np.asarray(ySplit))

    def PrintTree(self):
        self.printNode(self.root, "")

    def printNode(self, node, spacing):
        if node == None:
            return
        if node.Feature == None:
            print(spacing + str(node.Value))
            return
        
        print(spacing + "Feature ",node.Feature,":")           
        newSpacing = spacing + "     "
        print(spacing,"  >= 0.5")
        self.printNode(node.LeftNode, newSpacing)
        print(spacing,"  <  0.5")
        self.printNode(node.RightNode, newSpacing)

class DecisionTreeNode(object):
    
    def __init__(self, leftNode, RightNode, idx, value):
        self.LeftNode = leftNode
        self.RightNode = RightNode
        self.Feature = idx
        self.Value = value

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