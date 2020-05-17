import numpy as np
import pdb
import os

from ELL.markov.remotes import *
from ELL.util import *
import time
from utility import *
from statsPython.f import *

def markov(func,wald,lamEllByK,ellGrid,offDiag,numCores):
    t0=time.time()
    memory('markov')
        
    numTraits=wald.shape[1]
    
    stat,mins=func(wald,numCores)
    offDiagVec=ro.FloatVector(tuple(offDiag))
    
    b_markov=bufCreate('markov',wald.shape)
    d=lamEllByK.shape[1]
    
    reps=len(wald)
    
    print('max (grid,stat) ({},{}), min (grid,stat) ({},{})'.format(np.max(ellGrid),np.max(wald),np.min(ellGrid),np.min(wald)))
    
    wald=np.clip(wald,np.min(ellGrid),np.max(ellGrid))
    
    t1=time.time()
    
    pids=[]
    b_markov=bufCreate('markov',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(markovHelp,repRange,b_markov,stat,lamEllByK,ellGrid,d,numTraits,offDiagVec)]

    for pid in pids:
        os.waitpid(0, 0)
    
    pvals=bufClose(b_markov)

    memory('markov')
    t2=time.time()
    
    log('{} : {} snps, {} min/snp'.format('markov',reps,((t1-t0)+numCores*(t2-t1))/(60*reps)))

    return(pvals)
