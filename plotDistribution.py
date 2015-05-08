#!/usr/bin/python

import sys, io
from scipy.io.arff import loadarff
import scipy.spatial.distance as pyDistance
import random
import numpy as np
import pylab as plt
import math

bins = 100

#weight vectors - even machines 9 features
#mll                =     -13.5988 
#mjj                =       2.9715 
#met_phi_centrality =      -0.0794 
#mt                 =     -12.3976 
#DYjj               =       2.8055 
#sumMlj             =       3.5845 
#dphill             =      -1.1754 
#contOLV            =      -1.1174 
#ptTotal            =      -2.7679 
#bias               =       0.2625

#weight vetors - odd machines 9 features
mll                = -12.6961
mjj                = 2.9721
met_phi_centrality = -0.3015
mt                 = -10.3634
DYjj               = 2.4768
sumMlj             = 5.0838
dphill             = -1.1921
contOLV            = -1.1194
ptTotal            = -4.3755
bias               = -0.1084

#weight vectors - even machines 31 features
#MET                = -0.7428  
#MET_phi            =  0.0553  
#leadingLep_E       =  0.0854  
#leadingLep_px      =  0.193   
#leadingLep_pz      = -0.6703  
#subleadingLep_E    = -0.7327  
#subleadingLep_px   =  5.6579  
#subleadingLep_py   = -0.1106  
#subleadingLep_pz   =  0.6753  
#FW1                = -0.7999  
#FW2                = -0.028   
#FW3                = -0.2165  
#FW4                = -0.1318  
#FW5                =  0.4356  
#jet_et_total       =  1.2679  
#jet_energy_total   =  1.9876  
#jet_px             = -3.2809  
#jet_py             = -0.3206  
#jet_pz             =  0.1263  
#HT                 =  1.3328  
#EV2                =  0.1717  
#EV1                = -0.1017  
#mll                =-13.6651  
#mjj                =  2.0276  
#met-phi_centrality = -0.2628  
#mt                 =-10.276   
#DYjj               =  2.9932  
#sumMlj             = -0.6136  
#dphill             = -0.455   
#contOLV            = -0.9065  
#ptTotal            = -3.4452  
#bias               =  2.3427      
                    
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
        myplt = plt.figure(1)
        plt1 = plt.subplot(211)
        #plt1.set_xlabel('Distance From SVM Hyperplane')
        plt1.set_ylabel('Frequency')
        #plt1.set_title('Distance Of Test Data To Trained SVM Hyperplane')
        signalHistInfo = plt.hist(x, bins, alpha=0.5, histtype='stepfilled', label='x')
        backgroundHistInfo = plt.hist(y, bins, alpha=0.5, histtype='stepfilled', label='y')
        plt.axis([-2, 1, 0, 3000])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.4)
        stackedHistBins, bin_edges, patches = plt.hist((x,y), bins, alpha=1, label='x', histtype='barstacked')
        cutValues = getCutInfo(stackedHistBins)

        #plot cut graph
        plt2 = plt.subplot(212)
        #plt2.set_xlabel('Cut Location')
        plt2.set_ylabel('Signal / sqrt(background)')
        plt2.set_xlabel('Distance From SVM Hyperplane')

        bin_edges = np.delete(bin_edges, -1)
        plt.axis([-2, 1, 0, 55])
        plt.plot(bin_edges,cutValues)
        plt.savefig('singal-to-noise-9Even.png')
        plt.show()

def calculateDistances(data):

        x = list()
        newData = removeLabels(data)
        
        #for 9 features 
        #plane = (mll, mjj, met_phi_centrality, mt, DYjj, sumMlj, dphill, 
        #          contOLV, ptTotal)

        #for 31 features
        plane = (MET, MET_phi, leadingLep_E, leadingLep_px, leadingLep_pz,
                 subleadingLep_E, subleadingLep_px, subleadingLep_py, 
                 subleadingLep_pz, FW1, FW2, FW3, FW4, FW5, jet_et_total, 
                 jet_energy_total, jet_px, jet_py, jet_pz, HT, EV2, EV1, mll, 
                 mjj, met_phi_centrality, mt, DYjj, sumMlj, dphill, contOLV, 
                 ptTotal)

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
