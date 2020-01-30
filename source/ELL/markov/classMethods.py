import ray
import numpy as np
import pdb

from ELL.markov.remotes import *
from ELL.util import memory

def markov(self,ell):
    memory('markov')

    numCores=self.numCores
    dList=self.dList
    
    r_ellStats=ray.put(ell)
    r_lamEllByK=self.r_lamEllByK
    N=self.N
    
    r_offDiagVec=ray.put(ro.FloatVector(tuple(ray.get(self.r_offDiagVec))),weakref=True)
    
    r_markov=ray.put(np.zeros(ell.shape),weakref=True)
    r_ellGrid=ray.put(self.ellGrid)
    
    reps=len(ell)
    
    objectIds=[]
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
            
        objectIds+=[markovHelp.remote(ray.put(repRange,weakref=True),r_markov,r_ellStats,r_lamEllByK,r_ellGrid,dList,N,r_offDiagVec)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))

    memory('markov')

    return(ray.get(r_markov))
