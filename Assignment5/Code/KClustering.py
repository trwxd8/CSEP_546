import random as rand
import math
import numpy as np
class KMeansClustering(object):
    def __init__(self, k, seed):
        rand.seed(seed)
        self.K = k
        #self.centroids = np.empty((k, 2))
        #for i in range(k):
        #    self.centroids[i][0] = rand.random()
        #    self.centroids[i][1] = rand.random()
        self.centroids = np.array([[0.16983328, 0.21771349],
                                      [0.24487227, 0.3465653],
                                      [0.29945147, 0.48617321],
                                      [0.36830959, 0.68929934]])
        print(self.centroids)

    def formCluster(self, iterations, x):
        for i in range(iterations):
            self.fitCentroids(x)
            print("centroids after ",i)
            print(self.centroids)
            input("Test")

    def FindClosestImage(self, x, xRaw):
        for i in range(self.K):
            closestPic = -1
            closestDistance = 10 
            closestImage = ''
            currCentroid = self.centroids[i]
            xLen = len(x)
            for idx in range(xLen):
                example = x[idx]
                dist = self.distance(currCentroid, example)
                if dist < closestDistance:
                    closestPic = example
                    closestDistance = dist
                    closestImage = xRaw[idx]
            print("Closest to ",i," is ",closestPic," file name:", closestImage)

    def fitCentroids(self, x):
        centroidDatasets = []
        for i in range(self.K):
            centroidDatasets.append([])

        for example in x:
            #print(example)
            closestCentroid = -1
            closestDistance = 10
            for i in range(self.K):
                dist = self.distance(self.centroids[i], example)
                if dist < closestDistance:
                    closestCentroid = i
                    closestDistance = dist
            #print("Best idx:",closestCentroid)
            centroidDatasets[closestCentroid].append(example)
            print(example[0]," ", example[1]," ", closestCentroid)
        for i in range(self.K):
            print(len(centroidDatasets[i]))
            self.loss(centroidDatasets[i], i)

    def distance(self, centroid, sample):
        d1 = sample[0] - centroid[0]
        d2 = sample[1] - centroid[1]

        return math.sqrt(d1**2 + d2**2)

    def loss(self, x, currIdx):
        meanLoc = [0,0]
        cnt = len(x)
        for sample in x:
            meanLoc[0] += sample[0]
            meanLoc[1] += sample[1]
        meanLoc[0] /= cnt * 1.0
        meanLoc[1] /= cnt * 1.0
        self.centroids[currIdx] = meanLoc

class KNearestNeightbors(object):
    def __init__(self, k):
        pass

class Index(object):
    def __init__(self, x, y):
        self.X = x
        self.Y = y