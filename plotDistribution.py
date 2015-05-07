#!/usr/bin/python

import sys, io
from scipy.io.arff import loadarff
import scipy.spatial.distance as pyDistance
import random
import numpy as np
import pylab as plt
import math

bins = 100

#weight vectors
mll = -12.6961
mjj = 2.9721
met_phi_centrality = -0.3015
mt = -10.3634
DYjj = 2.4768
sumMlj = 5.0838
dphill = -1.1921
contOLV = -1.1194
ptTotal = -4.3755
bias = -0.1084
def main():
        if len(sys.argv) < 2:
                print "Exception: Too few arguments"
                print "usage: python plotDistances.py [FILE_PATH]"
                exit(0)


        elif len(sys.argv) > 3:
                print "Exception: Too many arguments"
                print "usage: python plotDistances.py [FILE_PATH]"
                exit(0)

        else:
                signalData = read_arff(sys.argv[1])
                backgroundData = read_arff(sys.argv[2])
                plotDistances(signalData, backgroundData)

        exit(1)

def plotDistances(signal, background):


        x = calculateDistances(signal)
        y = calculateDistances(background)

        #plot histogram of distances from hyperplane
        plt.figure(1)
        plt.subplot(211)
        plt.xlabel('Distance From SVM Hyperplane')
        plt.ylabel('Frequency')
        plt.title('Distance Of Test Data To Trained SVM Hyperplane')
        signalHistInfo = plt.hist(x, bins, alpha=0.5, histtype='stepfilled', label='x')
        backgroundHistInfo = plt.hist(y, bins, alpha=0.5, histtype='stepfilled', label='y')
        plt.axis([-2, 1, 0, 3000])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.4)
        stackedHistBins, bin_edges, patches = plt.hist((x,y), bins, alpha=1, label='x', histtype='barstacked')
        cutValues = getCutInfo(stackedHistBins)

        #plot cut graph
        plt.subplot(212)
        bin_edges = np.delete(bin_edges, -1)
        plt.axis([-2, 1, 0, 55])
        plt.plot(bin_edges,cutValues)
        plt.ylabel('cuts')
        plt.savefig('HiggsGraphs.png')
        plt.show()

def calculateDistances(data):

        x = list()
        newData = removeLabels(data)

        plane = (mll,mjj,met_phi_centrality,mt,DYjj,sumMlj,dphill,contOLV,ptTotal)
        for i in range(len(data)):
                v = np.array(plane)
                x.append(np.dot(newData[i], plane)/np.linalg.norm(v) - bias)

        return x


def removeLabels(data):
        newData = list()
        for point in data:
                newPoint = list()
                for i in range(0, (len(point) - 1)):
                        newPoint.append(point[i])
                newData.append(newPoint)
        return newData


'''
Takes in an array of size two that contains two arrays.
These correspond to the bins of signal and background for a
stacked histogram. What is returned is data in an array that is
s/b^(0.5)
'''
def getCutInfo(bins):
        signif = list()
        sigBins = bins[0].tolist()
        backBins = bins[1].tolist()

        for i in range(len(sigBins)):
                backSum = 0
                sigSum = 0
                for j in range(i, len(sigBins)):
                        sigSum += sigBins[j]
                        backSum += backBins[j]
                signifNum = sigSum/math.sqrt(backSum)
                signif.append(signifNum)
        return signif

#read in and parse arff file and store values in global vars attributes and data
def read_arff(file):
        loadData, loadMeta = loadarff(file)
        attributes  = list(loadMeta)
        data  = list(loadData)

        return data





if __name__ == "__main__":
        exit(main())
