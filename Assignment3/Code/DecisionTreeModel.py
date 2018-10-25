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
        self.root = self.growTree(np_x, np_y)
        pass

    def predict(self, x):
        return 0

    def informationGain(self):
        pass

    def growTree(self, x, y):
        value, counts = np.unique(y, return_counts=True)
        instanceDict = dict(zip(value, counts))
        print(instanceDict)

        zeroCnt = instanceDict[0]
        oneCnt = instanceDict[1]
        cnt = len(y)

        print("curr split: x-",x.shape," y-",y.shape)

        #If below minimum threshold, return value that occurs most 
        if cnt < self.minSplits:
            return DecisionTreeNode(Null, Null, 0 if zeroCnt > oneCnt else 1)
        
        #If all values are the same, return node of value
        elif oneCnt == 0:
            return DecisionTreeNode(Null, Null, 0)
        elif zeroCnt == 0:
            return DecisionTreeNode(Null, Null, 1)

        #Pick feature to split on 
        else:
            xIdx = self.PickBestSplit(x, y)
            print("Best index is ",xIdx)

            #Split datasets on value for feature
            (xSplit0, ySplit0) = self.SplitByFeature(x, y, xIdx, 0)
            (xSplit1, ySplit1) = self.SplitByFeature(x, y, xIdx, 1)
            print("0 split: x-",xSplit0.shape," y-",ySplit0.shape)
            print("1 split: x-",xSplit1.shape," y-",ySplit1.shape)           
            input("Enter input:")
        return DecisionTreeNode(self.growTree(xSplit0, ySplit0), self.growTree(xSplit1, ySplit1), xIdx)

    def PickBestSplit(self, x, y):

        infoGains = []
        cntFeatures = len(x[0])

        #Calculate information gain for each feature
        for idx in range(cntFeatures):
            print("finding gain for ",idx,"/",cntFeatures)
            infoGains.append(self.InformationGains(x, y, idx))
        print(infoGains)
        return np.argmax(infoGains)

    def InformationGains(self, x, y, idx):
        print("Info gain for ",idx)
        return self.Entropy(y) - self.Loss(x, idx)

    def Entropy(self, y):
        value, counts = np.unique(y, return_counts=True)
        instanceDict = dict(zip(value, counts))

        sumEntropy = 0
        cnt = np.size(y)

        for currKey in instanceDict:
            print("entropy for ",currKey)
            currProb = (instanceDict[currKey] + 1) / (cnt + 2)
            sumEntropy += currProb * math.log2(currProb)

        return -sumEntropy

    def Loss(self, x, idx):
        print(x.shape)
        xTransposed = np.transpose(x)
        print(xTransposed.shape)
        xIdxDataset = xTransposed[idx]
        value, counts = np.unique(xIdxDataset, return_counts=True)
        instanceDict = dict(zip(value, counts))

        lenSet = np.size(xIdxDataset)
        lossSum = 0
        #Add loss of 
        for currKey in instanceDict:
            print("Loss for ",currKey)
            lossSum += self.Entropy(xIdxDataset) * instanceDict[currKey]
        return lossSum / lenSet

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

class DecisionTreeNode(object):
    
    def __init__(self, leftNode, RightNode, value):
        self.LeftNode = leftNode
        self.RightNode = RightNode
        self.Value = value
