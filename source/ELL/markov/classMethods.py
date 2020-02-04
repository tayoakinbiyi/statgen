import ray
import numpy as np
import pdb
import os

from ELL.markov.remotes import *
from ELL.util import memory

def markov(self,ell,offDiag):
    memory('markov')

    numCores=self.numCores
    dList=self.dList
    
    r_ellStats=ray.put(ell)
    r_lamEllByK=ray.put(self.lamEllByK)
    N=self.N
    
    r_offDiagVec=ray.put(ro.FloatVector(tuple(offDiag)),weakref=True)
    
    r_markov=ray.put(np.zeros(ell.shape),weakref=True)
    r_ellGrid=ray.put(self.ellGrid)
    
    reps=len(ell)
    
    print('markov ({}): r_lamEllByK {}, get(r_lamEllByK) {}'.format(os.getpid(), id(r_lamEllByK),id(ray.get(r_lamEllByK))))

    objectIds=[]
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
            
        objectIds+=[markovHelp.remote(ray.put(repRange,weakref=True),r_markov,r_ellStats,r_lamEllByK,r_ellGrid,dList,N,r_offDiagVec)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))

    memory('markov')

    return(ray.get(r_markov))
