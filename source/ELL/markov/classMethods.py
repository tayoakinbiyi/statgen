import numpy as np
import pdb
import os

from ELL.markov.remotes import *
from ELL.util import *

def markov(self,ellStats):
    memory('markov')
    
    pdb.set_trace()
    numCores=self.numCores
    dList=self.dList
    offDiag=self.offDiag
    
    lamEllByK=self.lamEllByK
    N=self.N
    
    offDiagVec=ro.FloatVector(tuple(offDiag))
    
    b_markov=bufCreate('markov',ellStats.shape)
    ellGrid=self.ellGrid
    
    reps=len(ellStats)
    
    pids=[]
    b_markov={}
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        b_markov[core]=bufCreate('markov-'+str(core),[len(repRange),ellStats.shape[1]])
        pids+=[remote(markovHelp,repRange,b_markov[core],ellStats,lamEllByK,ellGrid,dList,N,offDiagVec)]

    for pid in pids:
        os.waitpid(0, 0)
    
    pvals=np.concatenate([b_markov[core][0] for core in range(len(b_markov))],axis=0)
    
    for core in b_markov:
        bufClose(b_markov[core])

    memory('markov')

    return(pvals)
