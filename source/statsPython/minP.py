import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *
from multiprocessing import cpu_count
from ELL.util import *

def minP(pi,numCores):
    t0=time.time()

    reps=pi.shape[0]
    
    pids=[]
    b_minP=bufCreate('minP',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(minPHelp,pi[repRange],b_minP,repRange)]

    for pid in pids:
        os.waitpid(0, 0)
        
    ans=bufClose(b_minP)
    
    t1=time.time()
        
    return(ans,numCores*(t1-t0)/(60))

def minPHelp(pi,b_minP,repRange):
    b_minP[0][repRange]=np.min(pi,axis=1)
    b_minP[1].flush()
    
    return()
