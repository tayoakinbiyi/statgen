import ray
import numpy as np
import pdb
import time
import psutil

import ELL.fit.classMethods as fitClass
import ELL.score.classMethods as scoreClass
import ELL.monteCarlo.classMethods as monteCarloClass
import ELL.markov.classMethods as markovClass

from ELL.util import nCr

class ell:
    
    def __init__(self,
                 offDiagVec,
                 N,
                 dList,
                 reportMem=False
                ):
        ray.init(object_store_memory=10e9)

        self.N=N
        self.r_offDiagVec=ray.put(offDiagVec)
        self.r_nCr=ray.put(nCr(N))
        self.r_offDiagMeans=ray.put(np.array([np.mean(offDiagVec), np.mean(offDiagVec**2), np.mean(offDiagVec**3),
                np.mean(offDiagVec**4),np.mean(offDiagVec**5), np.mean(offDiagVec**6), np.mean(offDiagVec**7),
                np.mean(offDiagVec**8),np.mean(offDiagVec**9), np.mean(offDiagVec**10)]))
        self.dList=dList.astype(int)
        
        self.numCores=int(ray.cluster_resources()['CPU'])
        self.scoreDone=False
        self.reportMem=reportMem
        
        self.firstMem=True
        
        return

    fit=fitClass.fit
    makeBins=fitClass.makeBins
    minMaxLamPerKInitial = fitClass.minMaxLamPerKInitial
    minMaxLamPerKFinal = fitClass.minMaxLamPerKFinal
    minMaxKPerBin = fitClass.minMaxKPerBin 
    callGetGamma = fitClass.callGetGamma
    callEllPerBinPerK=fitClass.callEllPerBinPerK
    callLamEllByK = fitClass.callLamEllByK
    
    score=scoreClass.score

    markov=markovClass.markov
    
    monteCarlo=monteCarloClass.monteCarlo
