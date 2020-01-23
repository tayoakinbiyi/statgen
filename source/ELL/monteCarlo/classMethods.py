import ray
import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb

def monteCarlo(self,numReps,L):
    assert self.scoreDone
    
    memory('monteCarlo')

    dList=self.dList
    reps=self.reps
    N=self.N

    r_ellStats=self.r_ellStats
    ellStats=ray.get(r_ellStats)
    
    r_offDiagVec=self.r_offDiagVec
    
    monteCarlo=np.zeros([len(ellStats),len(dList)])

    z=-np.sort(-np.abs(np.matmul(norm.rvs(size=[int(numReps),N]),L.T)))   
    _=self.score(z)
    
    del z,_
    refStats=ray.get(self.r_ellStats)
        
    for dInd in range(len(dList)):
        sortOrd=np.argsort(ellStats[:,dInd],axis=0)
        monteCarlo[sortOrd,dInd]=np.searchsorted(np.sort(refStats[:,dInd]),ellStats[sortOrd,dInd])/numReps
        
    self.reps=reps
    self.r_ellStats=r_ellStats
    
    del refStats
    
    memory('monteCarlo')

    return(monteCarlo)
