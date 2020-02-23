import numpy as np
import ray
from scipy.stats import norm
import pdb
import warnings
import os
import time

#from ELL.fit.nonClassMethods import *

@ray.remote
def f(N,lam,minK,maxK,s_nCr,s_offDiagMeans):
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
    
def lamEllByK(core,binRange,N,rightEdgePerBin,minKPerBin,maxKPerBin,nCr,lamEllByK,ellGrid,offDiagMeans):
    print('lamEllByK.remote ({}): lamEllByK {}'.format(os.getpid(), id(lamEllByK)))
    rightEdgePerBin=rightEdgePerBin[binRange]
    minKPerBin=minKPerBin[binRange]
    maxKPerBin=maxKPerBin[binRange]
    res=[]
    for Bin in range(len(binRange)):  
        t0=time.time()
        kList=np.arange(minKPerBin[Bin],maxKPerBin[Bin]+1).astype(int)
        t1=time.time()
        fval=ray.get(f.remote(N,rightEdgePerBin[Bin],minKPerBin[Bin],maxKPerBin[Bin],nCr,offDiagMeans))  
        t2=time.time()
        gridLoc=np.searchsorted(ellGrid,fval[::-1])[::-1]
        t3=time.time()
        upd=(gridLoc<len(ellGrid))        
        t4=time.time()
        lamEllByK[gridLoc[upd],kList[upd]]=np.maximum(rightEdgePerBin[Bin],lamEllByK[gridLoc[upd],kList[upd]])
        t5=time.time()
        res+=[[t1-t0,t2-t1,t3-t2,t4-t3,t5-t4]]
        if core==0 and Bin%int(len(binRange)/5)==0:
            print('core {} lamEllByK {}'.format(core,id(lamEllByK)),flush=True)
    print(core,np.mean(np.array(res),axis=0))
    
    return()
