from scipy.stats import norm
import ray
import sys
import pdb
import numpy as np
import time
import psutil
import os

from ELL.fit.remotes import *
from ELL.fit.nonClassMethods import *
from ELL.util import memory

def fit(self,initialNumLamPoints,finalNumLamPoints,minEllPValue,numEllPoints):
    self.finalNumLamPoints=finalNumLamPoints
    self.initialNumLamPoints=initialNumLamPoints
    ellGrid=np.exp(np.linspace(np.log(minEllPValue),np.log(.999),numEllPoints))
    
    memory('fit')
    
    self.minMaxLamPerKInitial()
    self.minMaxKPerBin()
    self.callGetGamma()
    pdb.set_trace()
    self.callEllPerBinPerK()
    self.callLamEllByK(ellGrid[[0,-1]])

    self.minMaxLamPerKFinal()    
    self.minMaxKPerBin()
    self.callGetGamma()
    self.callEllPerBinPerK()
    self.callLamEllByK(ellGrid)
    
    return()

def makeBins(self,numSteps,start=1e-40,stop=1):    
    bins=np.exp(np.linspace(np.log(start),np.log(stop),numSteps+1))
    self.r_leftEdgePerBin=ray.put(bins[0:-1])
    self.r_rightEdgePerBin=ray.put(bins[1:])

    memory('makeBins')
    
    return()

def minMaxLamPerKInitial(self):
    maxD=self.dList[-1]
    initialNumLamPoints=self.initialNumLamPoints

    self.r_minLamPerK=ray.put(np.zeros(maxD))
    self.r_maxLamPerK=ray.put(np.ones(maxD))
    
    self.makeBins(initialNumLamPoints)

    memory('minMaxLamPerKInitial')

    return()

def minMaxLamPerKFinal(self):    
    finalNumLamPoints=self.finalNumLamPoints
    lamEllByK=ray.get(self.r_lamEllByK)
    
    self.r_minLamPerK=ray.put(lamEllByK[0])
    self.r_maxLamPerK=ray.put(lamEllByK[-1])
        
    self.makeBins(finalNumLamPoints,lamEllByK[0,0],lamEllByK[-1,-1])

    memory('minMaxLamPerKFinal')
    
    return()
    
def minMaxKPerBin(self):
    minLamPerK=ray.get(self.r_minLamPerK)
    maxLamPerK=ray.get(self.r_maxLamPerK)
    
    leftEdgePerBin=ray.get(self.r_leftEdgePerBin)
    rightEdgePerBin=ray.get(self.r_rightEdgePerBin)
    
    maxBin=len(rightEdgePerBin)-1
    
    minBinPerK=np.clip(np.searchsorted(rightEdgePerBin,minLamPerK),0,maxBin)
    minBinPerK[0]=0

    maxBinPerK=np.clip(np.searchsorted(rightEdgePerBin,maxLamPerK),0,maxBin)
    
    numBinPerK=np.cumsum(maxBinPerK-minBinPerK+1)
    cumBinStartPerK=np.append(np.array([0]),numBinPerK[:-1])
            
    maxBin=int(max(maxBinPerK))

    self.r_minKPerBin=ray.put(minYPerXFromMaxXPerY(maxXPerY=maxBinPerK,minX=0))
    self.r_maxKPerBin=ray.put(maxYPerXFromMinXPerY(minXPerY=minBinPerK,maxX=maxBin))
    
    leftEdgePerBin=leftEdgePerBin[0:maxBin+1]
    rightEdgePerBin=rightEdgePerBin[0:maxBin+1]  
    
    midPointPerBin=(leftEdgePerBin+rightEdgePerBin)/2
    
    self.r_maxBinPerK=ray.put(maxBinPerK)
    self.r_minBinPerK=ray.put(minBinPerK)
    self.r_cumBinStartPerK=ray.put(cumBinStartPerK)
    self.r_rightEdgePerBin=ray.put(rightEdgePerBin)
    self.r_midPointPerBin=ray.put(midPointPerBin)
    self.size=np.sum(numBinPerK)

    memory('minMaxKPerBin')
    
    return()

def callGetGamma(self):
    numCores=self.numCores

    r_offDiagMeans=self.r_offDiagMeans
    r_midPointPerBin=self.r_midPointPerBin
    maxBin=len(ray.get(self.r_rightEdgePerBin))-1
    r_gammaPerBin=ray.put(np.empty(maxBin+1))
    
    objectIds=[]
    for core in range(numCores):
        binRange=np.clip(np.arange(core*int(np.ceil((maxBin+1)/numCores)),(core+1)*int(np.ceil((maxBin+1)/numCores))+1),0,maxBin)
        if len(binRange)==0:
            continue
            
        objectIds+=[getGamma.remote(ray.put(binRange),r_offDiagMeans,r_midPointPerBin,r_gammaPerBin)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))
    
    self.r_gammaPerBin=r_gammaPerBin

    memory('callGetGamma')
    
    return()
        
def callEllPerBinPerK(self):
    numCores=self.numCores
    N=self.N
    maxD=self.dList[-1]

    r_minKPerBin=self.r_minKPerBin
    r_maxKPerBin=self.r_maxKPerBin
    r_gammaPerBin=self.r_gammaPerBin
    r_midPointPerBin=self.r_midPointPerBin
    r_minBinPerK=self.r_minBinPerK
    r_cumBinStartPerK=self.r_cumBinStartPerK
    r_nCr=self.r_nCr

    maxBin=len(ray.get(self.r_rightEdgePerBin))-1
    r_ellPerBinPerK=ray.put(np.empty(self.size))

    objectIds=[]
    for core in range(numCores):
        binRange=np.clip(np.arange(core*int(np.ceil((maxBin+1)/numCores)),(core+1)*int(np.ceil((maxBin+1)/numCores))+1),0,maxBin)
        if len(binRange)==0:
            continue
        
        objectIds+=[ellPerBinPerK.remote(core,ray.put(binRange),N,r_midPointPerBin,r_minKPerBin,r_maxKPerBin,r_gammaPerBin,
            r_nCr,r_cumBinStartPerK,r_minBinPerK,r_ellPerBinPerK)]
        
        ready_ids, remaining_ids = ray.wait(objectIds[-1:], num_returns=1)
    
    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))
    
    self.r_midPointPerBin=None
    self.r_nCr=None
    self.r_gammaPerBin=None
    self.r_ellPerBinPerK=r_ellPerBinPerK
    self.r_minKPerBin=None
    self.r_maxKPerBin=None
    self.r_leftEdgePerBin=None
    self.r_midPointPerBin=None
   
    memory('callEllPerBinPerK')

    return()

def callLamEllByK(self,ellGrid):    
    numCores=self.numCores
    maxD=self.dList[-1]

    r_minBinPerK=self.r_minBinPerK
    r_maxBinPerK=self.r_maxBinPerK
    r_cumBinStartPerK=self.r_cumBinStartPerK
    r_rightEdgePerBin=self.r_rightEdgePerBin
    r_ellPerBinPerK=self.r_ellPerBinPerK

    r_lamEllByK=ray.put(np.full([len(ellGrid),maxD],np.nan))
    r_ellGrid=ray.put(ellGrid) 
    
    objectIds=[]
    for core in range(numCores):
        kRange=np.clip(np.arange(core*int(np.ceil(maxD/numCores)),(core+1)*int(np.ceil(maxD/numCores))+1),0,maxD)
        if len(kRange)==0:
            continue
        
        objectIds+=[lamEllByK.remote(ray.put(kRange),r_rightEdgePerBin,r_cumBinStartPerK,r_minBinPerK,r_maxBinPerK,r_ellPerBinPerK,
            r_lamEllByK,r_ellGrid)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))

    self.r_minBinPerK=None
    self.r_maxBinPerK=None
    self.r_rightEdgePerBin=None  
    self.r_ellPerBinPerK=None
    
    self.r_lamEllByK=r_lamEllByK
    self.r_ellGrid=r_ellGrid

    memory('callLamEllByK')

    return()
 