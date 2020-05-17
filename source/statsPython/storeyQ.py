import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *
from multiprocessing import cpu_count
from ELL.util import *

def storeyQ(d,wald,numCores):
    t0=time.time()
    
    pi=2*norm.sf(np.abs(wald))

    reps=pi.shape[0]
    
    pids=[]
    b_storeyQ=bufCreate('storeyQ',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(storeyQHelp,pi[repRange],d,b_storeyQ,repRange)]

    for pid in pids:
        os.waitpid(0, 0)
        
    ans=bufClose(b_storeyQ)
    
    t1=time.time()
        
    return(ans,numCores*(t1-t0)/(60))

def storeyQHelp(pi,d,b_storeyQ,repRange):
    b_storeyQ[0][repRange]=np.min(np.sort(pi*np.arange(1,pi.shape[1]+1).reshape(1,-1))[:,0:d],axis=1)
    b_storeyQ[1].flush()
    
    return()
