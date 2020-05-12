import numpy as np
import pdb
import os

from ELL.markov.remotes import *
from ELL.util import *
import time
from utility import *

def markov(self,ellStats):
    t0=time.time()
    memory('markov')
    
    numCores=self.numCores
    offDiag=self.offDiag
    
    lamEllByK=self.lamEllByK
    N=self.N
    
    offDiagVec=ro.FloatVector(tuple(offDiag))
    
    b_markov=bufCreate('markov',ellStats.shape)
    ellGrid=self.ellGrid
    d=self.d
    
    reps=len(ellStats)
    
    print('max (grid,stat) ({},{}), min (grid,stat) ({},{})'.format(
        np.max(ellGrid),np.max(ellStats),np.min(ellGrid),np.min(ellStats)))
    
    ellStats=np.clip(ellStats,np.min(ellGrid),np.max(ellGrid))
    
    pids=[]
    b_markov=bufCreate('markov',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(markovHelp,repRange,b_markov,ellStats,lamEllByK,ellGrid,d,N,offDiagVec)]

    for pid in pids:
        os.waitpid(0, 0)
    
    pvals=bufClose(b_markov)

    memory('markov')
    t1=time.time()
    
    log('{} : {} snps, {} min/snp'.format('markov',reps,(t1-t0)/(60*reps)))

    return(pvals)
