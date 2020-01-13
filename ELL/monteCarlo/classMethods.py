import ray
import numpy as np

def monteCarlo(self,numReps,L):
    assert self.scoreDone
    
    self.memory('monteCarlo')
 
    dList=self.dList
    reps=self.reps
    N=self.N

    ellStat=ray.get(self.ellStat)
    r_offDiagVec=self.r_offDiagVec
    
    monteCarlo=np.zeros([len(ell),len(self.dList)])

    z=-np.sort(-np.abs(np.matmul(norm.rvs(size=[numReps,N]),L.T)))   
    _=self.score(z)
    refStat=np.sort(ray.get(self.ellStat),axis=0)
    
    monteCarloMat=np.zeros([reps, len(dList)])
    
    for dInd in range(len(dList)):
        sortOrd=np.argsort(ell,axis=0)
        monteCarlo[sortOrd,dInd]=np.searchsorted(refStat[:,dInd],ellStat[sortOrd,dInd])/numReps
        
    self.memory('monteCarlo')

    return(monteCarloMat)
