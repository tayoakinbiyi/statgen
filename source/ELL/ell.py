import numpy as np
import pdb
import time
import psutil

import numpy as np
from ELL.util import *
from multiprocessing import cpu_count
from utility import *

from scipy.stats import norm
import sys
import time
import os

class ell:    
    def __init__(self,d,vZ,numCores=cpu_count()):    
        self.N=vZ.shape[1]
        self.nCr=nCr(vZ.shape[1])
        self.d=d
        self.numCores=numCores
        offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
        self.offDiag=offDiag
        self.offDiagMeans=np.array([np.mean(offDiag), np.mean(offDiag**2), np.mean(offDiag**3),
            np.mean(offDiag**4),np.mean(offDiag**5), np.mean(offDiag**6), np.mean(offDiag**7),
            np.mean(offDiag**8),np.mean(offDiag**9), np.mean(offDiag**10)])
        self.L=makeL(vZ)    

        self.lamEllByK=None

        memory('init')

        return

    def preCompute(self,ellStepSize,minVal):
        t0=time.time()

        numCores=self.numCores

        memory('start fit')

        N=self.N

        self.ellStepSize=ellStepSize

        ellGrid=geomBins(ellStepSize,minVal)

        self.minMaxLamPerKInitial()
        self.minMaxKPerBin()

        t1=time.time()
        self.callGetLamEllByK(ellGrid[[0,-1]])

        t2=time.time()
        self.minMaxLamPerKFinal()    
        self.minMaxKPerBin()

        t3=time.time()
        self.callGetLamEllByK(ellGrid)

        t4=time.time()
        self.ellGrid=ellGrid
        self.lamEllByK=bufClose(self.b_lamEllByK)

        t5=time.time()

        log('{} : {} min'.format('preCompute',((t1-t0)+numCores*(t2-t1)+(t3-t2)+numCores*(t4-t3)+(t5-t4))/(60)))

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

        memory('minMaxKPerBin')

        return()

    def callGetLamEllByK(self,ellGrid):
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

        memory('callGetLamEllByK')

        for pid in pids:
            os.waitpid(0, 0)

        print('num missing cells {} of {}'.format(np.sum(np.sum(b_lamEllByK[0]==0,axis=0)),np.prod(b_lamEllByK[0].shape)))

        del self.rightEdgePerBin
        del self.minKPerBin
        del self.maxKPerBin
        del self.ellGrid

        memory('callGetLamEllByK')

        return()

def F(N,lam,minK,maxK,nCr,offDiagMeans):
    assert 0<lam<1
    
    z=-norm.ppf(lam/2)

    He1 = z**2
    He3 = (z**3-3*z)**2
    He5 = (z**5-10*z**3+15*z)**2
    He7 = (z**7-21*z**5+105*z**3-105*z)**2
    He9 = (z**9-36*z**7+378*z**5-1260*z**3+945*z)**2
    
    odds = ( He1*offDiagMeans[1]/2 + He3*offDiagMeans[3]/24 + He5*offDiagMeans[5]/720 + He7*offDiagMeans[7]/40320 + 
            He9*offDiagMeans[9]/3628800 )
    
    x=4*norm.pdf(z)**2*odds    
    gamma=(x/(lam*(1-lam)-x))
    
    baseOne=np.append([0],np.cumsum(np.log(lam+gamma*np.arange(0,maxK))))
    baseTwo=np.cumsum(np.log(1-lam+gamma*np.arange(N)))[-(maxK+1):][::-1]
    baseThree=np.sum(np.log(1+gamma*np.arange(N)))
    baseCr=nCr[0:(int(maxK)+1)]

    Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
    
    return(1-np.cumsum(Pr)[minK:maxK+1])

def getLamEllByK(binRange,N,rightEdgePerBin,minKPerBin,maxKPerBin,nCr,b_lamEllByK,ellGrid,offDiagMeans):
    rightEdgePerBin=rightEdgePerBin[binRange]
    minKPerBin=minKPerBin[binRange]
    maxKPerBin=maxKPerBin[binRange]

    t0=time.time()
    for Bin in range(len(binRange)):  
        t1=time.time()
        if (t1-t0)/60>1:
            print('bin {} of {}'.format(Bin,len(binRange)),flush=True)
            t0=t1
        kList=np.arange(minKPerBin[Bin],maxKPerBin[Bin]+1).astype(int)
        fval=F(N,rightEdgePerBin[Bin],minKPerBin[Bin],maxKPerBin[Bin],nCr,offDiagMeans)
        gridLoc=np.searchsorted(ellGrid,fval[::-1])[::-1]
        upd=(gridLoc<len(ellGrid))        
        b_lamEllByK[0][gridLoc[upd],kList[upd]]=np.maximum(rightEdgePerBin[Bin],b_lamEllByK[0][gridLoc[upd],kList[upd]])
        b_lamEllByK[1].flush()
    
    return()

def minYPerXFromMaxXPerY(maxXPerY,minX):
    maxX=np.max(maxXPerY)
    uniqueMaxXPerY=np.sort(np.unique(maxXPerY))
    minYPer_uniqueMaxXPerY=np.searchsorted(maxXPerY,uniqueMaxXPerY,side='left')
    minYPerX=minYPer_uniqueMaxXPerY[np.searchsorted(uniqueMaxXPerY,np.arange(minX,maxX+1),side='left')].astype(int)
    
    return(minYPerX)

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

def geomBins(numSteps,minVal):
    maxVal=1-minVal
    zeta=np.power(minVal/maxVal,1/numSteps)
    bins=np.append(np.array([maxVal]),maxVal*np.power(zeta,np.arange(1,numSteps+1)))[::-1]
    return(bins)