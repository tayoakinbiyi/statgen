from scipy.stats import norm
import sys
import pdb
import numpy as np
import time
import psutil
import os

from ELL.fit.remotes import *
from ELL.util import *

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

def fit(self,numLamSteps0,numLamSteps1,numEllSteps,minEll,offDiag=None):
    if self.reportMem:
        memory('start fit')
    
    N=self.N
    if offDiag is None:
        offDiag=np.array([0]*int(N*(N-1)/2))
    
    self.offDiag=offDiag
    self.nCr=nCr(N)
    self.offDiagMeans=np.array([np.mean(offDiag), np.mean(offDiag**2), np.mean(offDiag**3),
        np.mean(offDiag**4),np.mean(offDiag**5), np.mean(offDiag**6), np.mean(offDiag**7),
        np.mean(offDiag**8),np.mean(offDiag**9), np.mean(offDiag**10)])
    
    ellGrid=geomBins(numEllSteps,minEll,1-minEll)

    self.numLamSteps0=numLamSteps0*self.N
    self.numLamSteps1=numLamSteps1*self.N
        
    self.minMaxLamPerKInitial(minEll)
    self.minMaxKPerBin()
    self.loopCallGetLamEllByK(ellGrid[[0,-1]])

    self.minMaxLamPerKFinal()    
    self.minMaxKPerBin()
    self.loopCallGetLamEllByK(ellGrid)
    
    self.ellGrid=ellGrid
    self.lamEllByK=bufClose(self.b_lamEllByK)
            
    return()

def minMaxLamPerKInitial(self,minEll):
    maxD=self.dList[-1]
    numLamSteps0=self.numLamSteps0

    self.minLamPerK=np.zeros(maxD)
    self.maxLamPerK=np.ones(maxD)
    
    x=1
    minF=1
    while minF>=minEll:
        x+=1
        minF=f(self.N,10**(-x),0,0,self.nCr,self.offDiagMeans)[0]
        
    self.minLam=10**(-x)
   
    lam=[0,1]
    while lam[1]-lam[0]>=1e-5:
        newLam=(lam[1]+lam[0])/2
        newF=f(self.N,newLam,maxD-1,maxD-1,self.nCr,self.offDiagMeans)[0]
        if newF<1-minEll:
            lam[0]=newLam
        else:
            lam[1]=newLam
        
    self.maxLam=lam[1]

    bins=geomBins(numLamSteps0,self.minLam,self.maxLam)
    
    self.rightEdgePerBin=bins

    if self.reportMem:
        memory('minMaxLamPerKInitial')

    return()

def minMaxLamPerKFinal(self):    
    numLamSteps1=self.numLamSteps1
    minLam=self.minLam
    maxLam=self.maxLam

    b_lamEllByK=self.b_lamEllByK[0]
    
    self.minLamPerK=b_lamEllByK[0]
    self.maxLamPerK=b_lamEllByK[-1]
        
    bins=geomBins(numLamSteps1,minLam,maxLam)

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

def callGetLamEllByK(self):
    numCores=self.numCores
    N=self.N
    maxBin=len(self.rightEdgePerBin)

    b_lamEllByK=self.b_lamEllByK
    minKPerBin=self.minKPerBin
    maxKPerBin=self.maxKPerBin
    rightEdgePerBin=self.rightEdgePerBin
    nCr=self.nCr
    ellGrid=self.ellGrid
    offDiagMeans=self.offDiagMeans
    
    pids=[]
    for core in range(numCores):
        binRange=np.arange(core*int(np.ceil(maxBin/numCores)),min(maxBin,(core+1)*int(np.ceil(maxBin/numCores))))
        if len(binRange)==0:
            continue
        
        pids+=[remote(getLamEllByK,core,binRange,N,rightEdgePerBin,minKPerBin,maxKPerBin,nCr,b_lamEllByK,ellGrid,offDiagMeans)]
        
    for pid in pids:
        os.waitpid(0, 0)
    
    return()

def loopCallGetLamEllByK(self,ellGrid):
    maxD=self.dList[-1]
    self.b_lamEllByK=bufCreate('lamEllByK',[len(ellGrid),maxD])
    b_lamEllByK=self.b_lamEllByK[0]

    self.ellGrid=ellGrid
    self.callGetLamEllByK()
        
    kHas0=np.where(np.sum(b_lamEllByK==0,axis=0)>0)[0]
    for k in kHas0:
        is0=np.where(b_lamEllByK[:,k]==0)[0]
        count=2
        while len(is0)>0:
            below=(is0-1)[~np.isin(is0-1,is0)]
            above=(is0+1)[~np.isin(is0+1,is0)]
            
            self.rightEdgePerBin=b_lamEllByK[below,k]+(b_lamEllByK[above,k]-b_lamEllByK[below,k])/count
            self.minKPerBin=np.array([k]*len(is0))
            self.maxKPerBin=np.array([k]*len(is0))
            
            self.callGetLamEllByK()

            is0=np.where(b_lamEllByK[:,k]==0)[0]
            count+=1
            
    del self.rightEdgePerBin
    del self.minKPerBin
    del self.maxKPerBin
    del self.ellGrid
    
    if self.reportMem:
        memory('callEllPerBinPerK')

    return()