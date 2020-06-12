import numpy as np
import pdb
import time
import psutil

import numpy as np
from multiprocessing import cpu_count
from utility import *

from scipy.stats import norm, beta
import sys
import time
import os

import math

import mmap
import subprocess
from scipy.special import betaln

class psi:    
    def __init__(self,calD,vZ,numLam,minEta,numCores,eps,maxIter,numHermite):    
        self.D=vZ.shape[1]
        self.dCr=dCrFunc(vZ.shape[1],calD)
        self.calD=calD
        self.numCores=numCores
        offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
        self.hermiteDiag=np.diag([(n%2==1)*np.mean(offDiag**(n+1))/math.factorial(n+1) for n in range(numHermite+1)])
        self.L=makeL(vZ)    
        self.numLam=numLam
        self.minEta=minEta
        self.eps=eps
        self.maxIter=maxIter

        return

    def compute(self):
        t0=time.time()

        print('minMaxLamPerD',flush=True)
        self.minMaxLamPerD()
        print('minMaxDPerBin',flush=True)
        self.minMaxDPerBin()
        
        t1=time.time()
        print('makePsi',flush=True)
        mins=self.makePsi()
        
        self.sortPsi()
        
        memory('psi')
        log('{} : {} min'.format('psi',(mins+(t1-t0)/60)))

        psiDF=bufClose(self.b_psi)
        psiDF['eta'][self.startPsiRowPerD]=2
        psiDF['eta'][self.endPsiRowPerD]=1
        psiDF=psiDF[psiDF['eta']>0]
        psiDF['eta'][psiDF['eta']==2]=0

        return(psiDF)

    def sortPsi(self): 
        calD=self.calD
        numCores=self.numCores

        startPsiRowPerD=self.startPsiRowPerD
        endPsiRowPerD=self.endPsiRowPerD
        
        b_psi=self.b_psi

        pids=[]
        for core in range(numCores):
            dRange=np.arange(core*int(np.ceil(calD/numCores)),min(calD,(core+1)*int(np.ceil(calD/numCores))))
            if len(dRange)==0:
                continue
            
            pids+=[remote(sortPsiHelp,dRange,startPsiRowPerD,endPsiRowPerD,b_psi)]
            
        for pid in pids:
            os.waitpid(0, 0)
        
        return()

    def minMaxLamPerD(self): 
        calD=self.calD
        minEta=self.minEta
        numLam=self.numLam
        D=self.D
        
        eps=self.eps
        maxIter=self.maxIter
        
        numCores=self.numCores
        hermiteDiag=self.hermiteDiag
        minEta=self.minEta
        dCr=self.dCr
        
        numLam=calD*numLam

        start=findLam(hermiteDiag,0,D,dCr,minEta,eps,maxIter,low=0,high=.1)[1]
        end=findLam(hermiteDiag,calD-1,D,dCr,1-minEta,eps,maxIter,low=calD/D,high=1)[0]
        
        lam=np.unique(geomBins(numLam,start,end))
        lam=lam[lam>0]
        
        b_minLamPerD=bufCreate('minLamPerD',[calD])
        b_maxLamPerD=bufCreate('maxLamPerD',[calD])
        
        numLam=len(lam)

        pids=[]
        for core in range(numCores):
            dRange=np.arange(core*int(np.ceil(calD/numCores)),min(calD,(core+1)*int(np.ceil(calD/numCores))))
            if len(dRange)==0:
                continue
            
            #minMaxLamPerDHelp(dRange,D,dCr,b_minLamPerD,b_maxLamPerD,minEta,hermiteDiag)
            pids+=[remote(minMaxLamPerDHelp,dRange,D,dCr,b_minLamPerD,b_maxLamPerD,minEta,hermiteDiag,eps,maxIter,start,end)]
            
        for pid in pids:
            os.waitpid(0, 0)
        
        self.minLamPerD=np.sort(bufClose(b_minLamPerD))
        self.maxLamPerD=np.sort(bufClose(b_maxLamPerD))
        self.lam=lam
        self.numLam=numLam

        return()

    def minMaxDPerBin(self):
        minLamPerD=self.minLamPerD
        maxLamPerD=self.maxLamPerD

        lam=self.lam
        numLam=len(lam)

        minBinPerD=np.clip(np.searchsorted(lam,minLamPerD),0,numLam-1)
        minBinPerD[0]=0

        maxBinPerD=np.clip(np.searchsorted(lam,maxLamPerD)-1,0,numLam-1)

        numLam=np.max(maxBinPerD)+1
        lam=lam[0:numLam]  

        self.minDPerBin=minYPerXFromMaxXPerY(maxXPerY=maxBinPerD,minX=0)
        self.maxDPerBin=maxYPerXFromMinXPerY(minXPerY=minBinPerD,maxX=numLam-1)

        self.minBinPerD=minBinPerD
        self.maxBinPerD=maxBinPerD
        self.lam=lam
        self.numLam=numLam

        return()

    def makePsi(self):
        t0=time.time()
        
        numCores=self.numCores
        D=self.D
        maxBin=len(self.lam)
        calD=self.calD
        hermiteDiag=self.hermiteDiag
        lam=self.lam
        dCr=self.dCr
        
        minBinPerD=self.minBinPerD
        maxBinPerD=self.maxBinPerD

        minDPerBin=self.minDPerBin
        maxDPerBin=self.maxDPerBin
        
        numLam=self.numLam
        
        psiLen=(maxBinPerD-minBinPerD+1)
        self.b_psi=bufCreate('psi',[np.sum(psiLen)],dtype=[('lam','float64'),('eta','float64')])
        b_psi=self.b_psi

        startPsiRowPerD=np.cumsum([0]+psiLen[:-1].tolist())
        endPsiRowPerD=np.cumsum(psiLen)-1
        
        self.startPsiRowPerD=startPsiRowPerD
        self.endPsiRowPerD=endPsiRowPerD

        psiLen=np.sum(psiLen)
        
        pids=[]
        t1=time.time()
        
        for core in range(numCores):
            binRange=np.arange(core*int(np.ceil(numLam/numCores)),min(numLam,(core+1)*int(np.ceil(numLam/numCores))))
            if len(binRange)==0:
                continue

            pids+=[remote(makePsiHelp,binRange,D,minDPerBin,maxDPerBin,startPsiRowPerD,b_psi,minBinPerD,
                dCr,lam,hermiteDiag)]
        
        for pid in pids:
            os.waitpid(0, 0)
            
        t2=time.time()
        
        return(((t1-t0)+numCores*(t2-t1))/60)

def minMaxLamPerDHelp(dRange,D,dCr,b_minLamPerD,b_maxLamPerD,minEta,hermiteDiag,eps,maxIter,low,high):
    for dInd in range(len(dRange)):
        lam=findLam(hermiteDiag,dRange[dInd],D,dCr,minEta,eps,maxIter,low,high)
        b_minLamPerD[0][dRange[dInd]]=lam[1]
        b_minLamPerD[1].flush()
        low=lam[0]

        lam=findLam(hermiteDiag,dRange[-dInd-1],D,dCr,1-minEta,eps,maxIter,low,high)
        b_maxLamPerD[0][dRange[-dInd-1]]=lam[0]
        b_maxLamPerD[1].flush()
        high=lam[1]

    return()

def findLam(hermiteDiag,d,D,dCr,eta,eps,maxIter,low,high):
    lam=[low,high]
    F=[getEta(D,low,d,d,dCr,hermiteDiag)[0],getEta(D,high,d,d,dCr,hermiteDiag)[0]]
    count=0
    
    while (F[1]-F[0])>eps and count<maxIter:
        newLam=(lam[1]+lam[0])/2
        newF=np.clip(getEta(D,newLam,d,d,dCr,hermiteDiag)[0],F[0],F[1])
        
        if newF<eta:
            lam[0]=newLam
            F[0]=newF
        else:
            lam[1]=newLam
            F[1]=newF
        count+=1
        
    return(lam)

def getEta(D,lam,minK,maxK,dCr,hermiteDiag):
    if lam==0:
        return(np.array([0]*(maxK-minK+1)))
    if lam==1:
        return(np.array([1]*(maxK-minK+1)))
              
    z=-norm.ppf(lam/2)

    x=4*norm.pdf(z)**2*np.polynomial.hermite_e.hermeval2d(z,z,hermiteDiag)    
    gamma=(x/(lam*(1-lam)-x))
    
    baseOne=np.append([0],np.cumsum(np.log(lam+gamma*np.arange(0,maxK))))
    baseTwo=np.cumsum(np.log(1-lam+gamma*np.arange(D)))[-(maxK+1):][::-1]
    baseThree=np.sum(np.log(1+gamma*np.arange(D)))
    baseCr=dCr[0:(int(maxK)+1)]

    Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
    
    return(np.clip(1-np.cumsum(Pr)[minK:maxK+1],0,1))


def makePsiHelp(binRange,D,minDPerBin,maxDPerBin,startPsiRowPerD,b_psi,minBinPerD,dCr,lam,hermiteDiag):
    for Bin in binRange:  
        dList=np.arange(minDPerBin[Bin],maxDPerBin[Bin]+1).astype(int)
        fval=getEta(D,lam[Bin],minDPerBin[Bin],maxDPerBin[Bin],dCr,hermiteDiag)
        loc=startPsiRowPerD[minDPerBin[Bin]:maxDPerBin[Bin]+1]+(Bin-minBinPerD[minDPerBin[Bin]:maxDPerBin[Bin]+1])
        b_psi[0]['lam'][loc]=lam[Bin]
        b_psi[0]['eta'][loc]=fval
        b_psi[1].flush()
    
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

def geomBins(numLam,minVal,maxVal):
    zeta=np.power(minVal/maxVal,1/numLam)
    bins=np.append(np.array([maxVal]),maxVal*np.power(zeta,np.arange(1,numLam+1)))[::-1]
    return(bins)

def dCrFunc(D,calD):
    dVec=np.arange(0,calD+1)
    ans=-betaln(1 + D - dVec, 1 + dVec) - np.log(D + 1)

    return(ans)
    
def sortPsiHelp(dRange,startPsiRowPerD,endPsiRowPerD,b_psi):
    for d in dRange:
        b_psi[0]['eta'][startPsiRowPerD[d]:endPsiRowPerD[d]+1]=np.sort(b_psi[0][
            'eta'][startPsiRowPerD[d]:endPsiRowPerD[d]+1])
        b_psi[1].flush()
        
    return()
    
