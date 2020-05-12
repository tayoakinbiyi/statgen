import numpy as np
from scipy.stats import norm
import pdb
import warnings
import os
import time

from ELL.F import *

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
