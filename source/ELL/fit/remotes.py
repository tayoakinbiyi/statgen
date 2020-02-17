import numpy as np
from scipy.stats import norm
import pdb
import warnings
import os
import time

def f(N,lam,minK,maxK,nCr,offDiagMeans):
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
    
def getLamEllByK(core,binRange,N,rightEdgePerBin,minKPerBin,maxKPerBin,nCr,b_lamEllByK,ellGrid,offDiagMeans):
    print(core,id(b_lamEllByK[0]))
    rightEdgePerBin=rightEdgePerBin[binRange]
    minKPerBin=minKPerBin[binRange]
    maxKPerBin=maxKPerBin[binRange]
    res=[]
    for Bin in range(len(binRange)):  
        kList=np.arange(minKPerBin[Bin],maxKPerBin[Bin]+1).astype(int)
        fval=f(N,rightEdgePerBin[Bin],minKPerBin[Bin],maxKPerBin[Bin],nCr,offDiagMeans)
        gridLoc=np.searchsorted(ellGrid,fval[::-1])[::-1]
        upd=(gridLoc<len(ellGrid))        
        b_lamEllByK[0][gridLoc[upd],kList[upd]]=np.maximum(rightEdgePerBin[Bin],b_lamEllByK[0][gridLoc[upd],kList[upd]])
        b_lamEllByK[1].flush()
    
    return()
