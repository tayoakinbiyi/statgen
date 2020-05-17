import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *
from multiprocessing import cpu_count
from ELL.util import *

def scoreTest(wald,numCores):
    t0=time.time()
    
    reps=wald.shape[0]
    
    pids=[]
    b_score=bufCreate('score',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(scoreTestHelp,wald[repRange],b_score,repRange)]

    for pid in pids:
        os.waitpid(0, 0)
        
    ans=bufClose(b_score)
    
    t1=time.time()
    
    return(ans,numCores*(t1-t0)/60)

def scoreTestHelp(wald,b_score,repRange):
    b_score[0][repRange]=np.sum(wald**2,axis=1)
    return()
