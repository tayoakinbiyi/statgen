import numpy as np
import ray
from scipy.stats import norm
import pdb

from ELL.fit.nonClassMethods import *

@ray.remote
def getGamma(binRange,offDiagMeans,midPointPerBin,gammaPerBin):
    midPointPerBin=midPointPerBin[binRange]
    z=-norm.ppf(midPointPerBin/2)

    He1 = z**2
    He3 = (z**3-3*z)**2
    He5 = (z**5-10*z**3+15*z)**2
    He7 = (z**7-21*z**5+105*z**3-105*z)**2
    He9 = (z**9-36*z**7+378*z**5-1260*z**3+945*z)**2
    
    odds = ( He1*offDiagMeans[1]/2 + He3*offDiagMeans[3]/24 + He5*offDiagMeans[5]/720 + He7*offDiagMeans[7]/40320 + 
            He9*offDiagMeans[9]/3628800 )
    
    x=4*norm.pdf(z)**2*odds    
    
    gammaPerBin[binRange]=(x/(midPointPerBin*(1-midPointPerBin)-x))
    
    return()

@ray.remote
def ellPerBinPerK(core,binRange,N,midPointPerBin,minKPerBin,maxKPerBin,gammaPerBin,nCr,cumBinStartPerK,minBinPerK,ellPerBinPerK):
    pdb.set_trace()
    midPointPerBin=midPointPerBin[binRange]
    minKPerBin=minKPerBin[binRange]
    maxKPerBin=maxKPerBin[binRange]
    gammaPerBin=gammaPerBin[binRange]
    
    for row in range(len(binRange)):
        rLam=midPointPerBin[row]
        rMin=minKPerBin[row]
        rMax=maxKPerBin[row]
        rGamma=gammaPerBin[row]

        baseOne=np.append([0],np.cumsum(np.log(rLam+rGamma*np.arange(0,rMax))))
        baseTwo=np.cumsum(np.log(1-rLam+rGamma*np.arange(N)))[-(rMax+1):][::-1]
        baseThree=np.sum(np.log(1+rGamma*np.arange(N)))
        baseCr=nCr[0:(int(rMax)+1)]

        Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
        kList=np.arange(rMin,rMax+1)

        ellPerBinPerK[cumBinStartPerK[kList]+(binRange[row]-minBinPerK[kList])]=1-np.cumsum(Pr)[rMin:rMax+1]
       
    return()

@ray.remote
def lamEllByK(kRange,rightEdgePerBin,cumBinStartPerK,minBinPerK,maxBinPerK,ellPerBinPerK,lamEllByK,ellGrid):
    for k in kRange:
        loc=np.searchsorted(ellPerBinPerK[cumBinStartPerK[k]:cumBinStartPerK[k]+(maxBinPerK[k]-minBinPerK[k]+1)],ellGrid)
        lamEllByK[:,k]=rightEdgePerBin[minBinPerK[k]+loc]
        
    return()     
