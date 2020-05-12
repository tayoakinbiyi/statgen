from scipy.stats import norm
import sys
import pdb
import numpy as np
import time
import psutil
import os

from ELL.preCompute.remotes import *
from ELL.util import *
import time
from utility import *

# requires:
# minY=0
# maxY=len(maxXPerY)-1
# maxX=max(maxXPerY)
# assumes
# minX=min(minXPerY)
def minYPerXFromMaxXPerY(maxXPerY,minX):
    maxX=np.max(maxXPerY)
    uniqueMaxXPerY=np.sort(np.unique(maxXPerY))
    minYPer_uniqueMaxXPerY=np.searchsorted(maxXPerY,uniqueMaxXPerY,side='left')
    minYPerX=minYPer_uniqueMaxXPerY[np.searchsorted(uniqueMaxXPerY,np.arange(minX,maxX+1),side='left')].astype(int)
    
    return(minYPerX)

# requires
# minY=0
# maxY=len(maxXPerY)-1
# minX=min(minXPerY)
# assumes
# maxX=max(maxXPerY)
def maxYPerXFromMinXPerY(minXPerY,maxX):
    minX=np.min(minXPerY)
    uniqueMinXPerY=np.sort(np.unique(minXPerY))    
    maxYPer_uniqueMinXPerY=np.searchsorted(minXPerY,uniqueMinXPerY,side='right')-1    
    maxYPerX=maxYPer_uniqueMinXPerY[np.searchsorted(uniqueMinXPerY,np.arange(minX,maxX+1),side='right')-1].astype(int)
    
    return(maxYPerX)

def logUnifBins(zeta,numSteps,minLam):
    assert zeta>=1
    a=zeta*(1-minLam)/(zeta-1)
    b=1-a
    return(a*np.power(10,np.linspace(-np.log10(zeta),0,numSteps+1))+b)

def geomBins(numSteps,minVal,maxVal):
    zeta=np.power(minVal/maxVal,1/numSteps)
    bins=np.append(np.array([maxVal]),maxVal*np.power(zeta,np.arange(1,numSteps+1)))[::-1]
    return(bins)

def preCompute(self,ellStepSize=1e3):
    t0=time.time()
    
    if self.reportMem:
        memory('start fit')
    
    N=self.N

    self.ellStepSize=ellStepSize
    
    ellGrid=geomBins(ellStepSize,1e-8,1-1e-8)
        
    self.minMaxLamPerKInitial()
    self.minMaxKPerBin()
    self.callGetLamEllByK(ellGrid[[0,-1]])

    self.minMaxLamPerKFinal()    
    self.minMaxKPerBin()
    self.callGetLamEllByK(ellGrid)
    
    self.ellGrid=ellGrid
    self.lamEllByK=bufClose(self.b_lamEllByK)
    
    t1=time.time()

    log('{} : {} min'.format('preCompute',(t1-t0)/(60)))
            
    return()

def minMaxLamPerKInitial(self):
    d=self.d

    self.minLamPerK=np.zeros(d)
    self.maxLamPerK=np.ones(d)
    
    minEll=1e-8
    
    x=1
    minF=1
    while minF>=minEll:
        x+=1
        minF=F(self.N,10**(-x),0,0,self.nCr,self.offDiagMeans)[0]
        
    self.minLam=10**(-x)
   
    lam=[0,1]
    while lam[1]-lam[0]>=minEll:
        newLam=(lam[1]+lam[0])/2
        newF=F(self.N,newLam,d-1,d-1,self.nCr,self.offDiagMeans)[0]
        if newF<1-minEll:
            lam[0]=newLam
        else:
            lam[1]=newLam
    
    self.maxLam=lam[1]
    
    bins=geomBins(1e3,self.minLam,self.maxLam)
    print('minMaxLamPerKInitial num bins {}'.format(len(bins)))
    
    self.rightEdgePerBin=bins

    if self.reportMem:
        memory('minMaxLamPerKInitial')

    return()

def minMaxLamPerKFinal(self):    
    ellStepSize=self.ellStepSize

    b_lamEllByK=self.b_lamEllByK[0]
    
    self.minLamPerK=b_lamEllByK[0]
    self.maxLamPerK=b_lamEllByK[-1]
        
    bins=geomBins(self.N*ellStepSize,self.minLam,self.maxLam)
    print('minMaxLamPerKFinal num bins {}'.format(len(bins)))

    self.rightEdgePerBin=bins

    if self.reportMem:
        memory('minMaxLamPerKFinal')
    
    return()
    
def minMaxKPerBin(self):
    minLamPerK=self.minLamPerK
    maxLamPerK=self.maxLamPerK
    
    rightEdgePerBin=self.rightEdgePerBin
    
    maxBin=len(rightEdgePerBin)-1
    
    minBinPerK=np.clip(np.searchsorted(rightEdgePerBin,minLamPerK)-1,0,maxBin)
    minBinPerK[0]=0

    maxBinPerK=np.clip(np.searchsorted(rightEdgePerBin,maxLamPerK),0,maxBin)
                
    maxBin=int(max(maxBinPerK))

    self.minKPerBin=minYPerXFromMaxXPerY(maxXPerY=maxBinPerK,minX=0)
    self.maxKPerBin=maxYPerXFromMinXPerY(minXPerY=minBinPerK,maxX=maxBin)
    
    rightEdgePerBin=rightEdgePerBin[0:maxBin+1]  
    
    self.maxBinPerK=maxBinPerK
    self.rightEdgePerBin=rightEdgePerBin
    self.maxBin=maxBin

    if self.reportMem:
        memory('minMaxKPerBin')
    
    return()

def callGetLamEllByK(self,ellGrid):
    if self.reportMem:
        memory('callGetLamEllByK')
    
    numCores=self.numCores
    N=self.N
    maxBin=len(self.rightEdgePerBin)
    d=self.d
    self.ellGrid=ellGrid

    self.b_lamEllByK=bufCreate('lamEllByK',[len(ellGrid),d])
    b_lamEllByK=self.b_lamEllByK

    minKPerBin=self.minKPerBin
    maxKPerBin=self.maxKPerBin
    rightEdgePerBin=self.rightEdgePerBin
    nCr=self.nCr
    offDiagMeans=self.offDiagMeans
    print('number of bins {}'.format(maxBin))
    
    pids=[]
    for core in range(numCores):
        binRange=np.arange(core*int(np.ceil(maxBin/numCores)),min(maxBin,(core+1)*int(np.ceil(maxBin/numCores))))
        if len(binRange)==0:
            continue
        
        pids+=[remote(getLamEllByK,binRange,N,rightEdgePerBin,minKPerBin,maxKPerBin,nCr,b_lamEllByK,ellGrid,offDiagMeans)]
        
    if self.reportMem:
        memory('callGetLamEllByK')

    for pid in pids:
        os.waitpid(0, 0)
       
    print('num missing cells {} of {}'.format(np.sum(np.sum(b_lamEllByK[0]==0,axis=0)),np.prod(b_lamEllByK[0].shape)))
    
    del self.rightEdgePerBin
    del self.minKPerBin
    del self.maxKPerBin
    del self.ellGrid
    
    if self.reportMem:
        memory('callGetLamEllByK')

    return()
