import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pdb
from ELL.util import *
from multiprocessing import cpu_count
import rpy2.robjects as ro
import time
from utility import *

def cpma(z,numCores=cpu_count()): 
    numSnps,numTraits=z.shape
    
    t0=time.time()
        
    t1=time.time()
    
    b_time=bufCreate('time',[numCores])
    b_res=bufCreate('res',[numSnps])
    pids=[]
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(cpmaHelp,core,z[snpRange],snpRange,b_time,b_res)]
        
    for pid in pids:
        os.waitpid(0, 0)
        
    return(bufClose(b_res),(np.sum(bufClose(b_time))+(t1-t0))/60)

def cpmaHelp(core,z,snpRange,b_time,b_res):  
    t0=time.time()
    np.savetxt('cpma-'+str(core),z,delimiter='\t')
    subprocess.call(['Rscript','../source/R/cpma.R','cpma-'+str(core)])
    b_res[0][snpRange]=np.loadtxt('cpma-'+str(core),delimiter='\t')
    b_res[1].flush()
    t1=time.time()
    
    b_time[0][core]=(t1-t0)
    b_time[1].flush()
    
    return()
