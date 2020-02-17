import numpy as np
import pdb
import os

from ELL.markov.remotes import *
from ELL.util import *

def markov(self,ellStats):
    memory('markov')

    numCores=self.numCores
    dList=self.dList
    offDiag=self.offDiag
    
    lamEllByK=self.lamEllByK
    N=self.N
    
    offDiagVec=ro.FloatVector(tuple(offDiag))
    
    b_markov=bufCreate('markov',ellStats.shape)
    ellGrid=self.ellGrid
    
    reps=len(ellStats)
    
    print('markov ({}): r_lamEllByK {}, get(r_lamEllByK) {}'.format(os.getpid(), id(lamEllByK),id(lamEllByK)))

    pids=[]
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
            
        pids+=[remote(markovHelp,repRange,b_markov,ellStats,lamEllByK,ellGrid,dList,N,offDiagVec)]

    for pid in pids:
        os.waitpid(0, 0)

    memory('markov')
    
    pvals=bufClose(b_markov)

    return(pvals)
