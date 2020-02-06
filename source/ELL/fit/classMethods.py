from scipy.stats import norm
import ray
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

def makeBins(zeta,numSteps,minLam):
    assert zeta>=1
    a=zeta*(1-minLam)/(zeta-1)
    b=1-a
    return(a*np.power(10,np.linspace(-np.log10(zeta),0,numSteps+1))+b)

def fit(self,initialNumLamPoints,finalNumLamPoints, numEllPoints,zeta,minEll):
    if self.reportMem:
        memory('start fit')

    N=self.N
    offDiag=self.offDiag

    self.r_nCr=ray.put(nCr(N))
    self.r_offDiagMeans=ray.put(np.array([np.mean(offDiag), np.mean(offDiag**2), np.mean(offDiag**3),
        np.mean(offDiag**4),np.mean(offDiag**5), np.mean(offDiag**6), np.mean(offDiag**7),
        np.mean(offDiag**8),np.mean(offDiag**9), np.mean(offDiag**10)]))
    
    ellGrid=makeBins(zeta,numEllPoints,2*minEll)/2
    ellGrid=np.append(ellGrid[:-1],1-ellGrid[::-1])

    self.finalNumLamPoints=finalNumLamPoints*self.N
    self.initialNumLamPoints=initialNumLamPoints*self.N
    self.ellZeta=zeta
    self.lamZeta=zeta
        
    self.minMaxLamPerKInitial(minEll)
    self.minMaxKPerBin()
    self.loopCallLamEllByK(ellGrid[[0,-1]])

    self.minMaxLamPerKFinal()    
    self.minMaxKPerBin()
    self.loopCallLamEllByK(ellGrid)
    
    self.ellGrid=ellGrid
        
    return()

def minMaxLamPerKInitial(self,ell):
    maxD=self.dList[-1]
    initialNumLamPoints=self.initialNumLamPoints

    self.r_minLamPerK=ray.put(np.zeros(maxD),weakref=True)
    self.r_maxLamPerK=ray.put(np.ones(maxD),weakref=True)
        
    x=1
    minF=1
    while minF>=ell:
        x+=1
        minF=ray.get(f.remote(self.N,10**(-x),0,0,self.r_nCr,self.r_offDiagMeans))[0]
        print(minF,x)

    self.minLam=10**(-x)
    
    bins=np.linspace(10**(-x), 1,initialNumLamPoints)
    bins=bins[(bins<1)&(bins>0)]
    
    self.r_rightEdgePerBin=ray.put(bins,weakref=True)

    if self.reportMem:
        memory('minMaxLamPerKInitial')

    return()

def minMaxLamPerKFinal(self):    
    finalNumLamPoints=self.finalNumLamPoints
    zeta=self.lamZeta
    minLam=self.minLam

    lamEllByK=ray.get(self.r_lamEllByK)
    
    self.r_minLamPerK=ray.put(lamEllByK[0],weakref=True)
    self.r_maxLamPerK=ray.put(lamEllByK[-1],weakref=True)
        
    bins=makeBins(zeta,finalNumLamPoints,minLam)[:-1]
    bins[0]=minLam

    self.r_rightEdgePerBin=ray.put(bins,weakref=True)

    if self.reportMem:
        memory('minMaxLamPerKFinal')
    
    return()
    
def minMaxKPerBin(self):
    minLamPerK=ray.get(self.r_minLamPerK)
    maxLamPerK=ray.get(self.r_maxLamPerK)
    
    rightEdgePerBin=ray.get(self.r_rightEdgePerBin)
    
    maxBin=len(rightEdgePerBin)-1
    
    minBinPerK=np.clip(np.searchsorted(rightEdgePerBin,minLamPerK)-1,0,maxBin)
    minBinPerK[0]=0

    maxBinPerK=np.clip(np.searchsorted(rightEdgePerBin,maxLamPerK),0,maxBin)
                
    maxBin=int(max(maxBinPerK))

    self.r_minKPerBin=ray.put(minYPerXFromMaxXPerY(maxXPerY=maxBinPerK,minX=0),weakref=True)
    self.r_maxKPerBin=ray.put(maxYPerXFromMinXPerY(minXPerY=minBinPerK,maxX=maxBin),weakref=True)
    
    rightEdgePerBin=rightEdgePerBin[0:maxBin+1]  
    
    self.r_maxBinPerK=ray.put(maxBinPerK,weakref=True)
    self.r_rightEdgePerBin=ray.put(rightEdgePerBin,weakref=True)
    self.maxBin=maxBin

    if self.reportMem:
        memory('minMaxKPerBin')
    
    return()

def callLamEllByK(self):
    numCores=self.numCores
    N=self.N
    maxBin=len(ray.get(self.r_rightEdgePerBin))

    r_lamEllByK=self.r_lamEllByK
    r_minKPerBin=self.r_minKPerBin
    r_maxKPerBin=self.r_maxKPerBin
    r_rightEdgePerBin=self.r_rightEdgePerBin
    r_nCr=self.r_nCr
    r_ellGrid=self.r_ellGrid
    r_offDiagMeans=self.r_offDiagMeans
    
    objectIds=[]
    print('callLamEllByK ({}): r_lamEllByK {}, get(r_lamEllByK) {}'.format(os.getpid(), id(r_lamEllByK),id(ray.get(r_lamEllByK))))
    for core in range(numCores):
        binRange=np.arange(core*int(np.ceil(maxBin/numCores)),min(maxBin,(core+1)*int(np.ceil(maxBin/numCores))))
        if len(binRange)==0:
            continue
        
        objectIds+=[lamEllByK.remote(core,ray.put(binRange,weakref=True),N,r_rightEdgePerBin,r_minKPerBin,
            r_maxKPerBin,r_nCr,r_lamEllByK,r_ellGrid,r_offDiagMeans)]
        
    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))
   
    return()

def loopCallLamEllByK(self,ellGrid):
    maxD=self.dList[-1]
    r_lamEllByK=ray.put(np.full([len(ellGrid),maxD],0.0),weakref=True)
    
    print('loopCallLamEllByK ({}): r_lamEllByK {}, get(r_lamEllByK) {}'.format(os.getpid(), id(r_lamEllByK),id(ray.get(r_lamEllByK))))

    self.r_lamEllByK=r_lamEllByK
    self.r_ellGrid=ray.put(ellGrid,weakref=True) 
    self.callLamEllByK()
        
    kHas0=np.where(np.sum(ray.get(r_lamEllByK)==0,axis=0)>0)[0]
    for k in kHas0:
        is0=np.where(ray.get(r_lamEllByK)[:,k]==0)[0]
        count=2
        while len(is0)>0:
            below=(is0-1)[~np.isin(is0-1,is0)]
            above=(is0+1)[~np.isin(is0+1,is0)]
            
            self.r_rightEdgePerBin=ray.put(ray.get(r_lamEllByK)[below,k]+(ray.get(r_lamEllByK)[above,k]-
                ray.get(r_lamEllByK)[below,k])/count)
            self.r_minKPerBin=ray.put(np.array([k]*len(is0)))
            self.r_maxKPerBin=ray.put(np.array([k]*len(is0)))
            
            self.callLamEllByK()

            is0=np.where(ray.get(r_lamEllByK)[:,k]==0)[0]
            count+=1
            
    del self.r_rightEdgePerBin
    del self.r_minKPerBin
    del self.r_maxKPerBin
    del self.r_ellGrid
    
    self.lamEllByK=ray.get(r_lamEllByK)
    
    if self.reportMem:
        memory('callEllPerBinPerK')

    return()