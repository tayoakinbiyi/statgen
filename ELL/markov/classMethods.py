import ray
import numpy as np

from ELL.markov.remotes import *

def markov(self):
    assert self.scoreDone
    
    self.memory('markov')

    numCores=self.numCores
    dList=self.dList
    
    r_ellStats=self.r_ellStats
    r_zEllByK=self.r_zEllByK
    r_ellGrid=self.r_ellGrid
    N=self.N
    
    r_offDiagVec=ray.put(ro.FloatVector(tuple(ray.get(self.r_offDiagVec))))
    reps=self.reps
    
    r_markov=ray.put(np.zeros([reps, len(dList)]))
    
    objectIds=[]
    for core in range(numCores):
        repRange=np.clip(np.arange(core*int(np.ceil(reps/numCores)),(core+1)*int(np.ceil(reps/numCores))),0,reps-1)

        if len(repRange)==0:
            continue
            
        objectIds+=[markovHelp.remote(ray.put(repRange),r_markov,r_stats,r_zEllByK,dList,N,r_offDiagVec)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))

    self.memory('markov')

    return(ray.get(r_markov))
