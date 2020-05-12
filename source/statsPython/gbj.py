import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pdb
from ELL.util import *
from multiprocessing import cpu_count
import rpy2.robjects as ro
import time
from utility import *

def gbj(func,z,numCores=cpu_count(),offDiag=None): 
    numSnps,numTraits=z.shape
    
    t0=time.time()
    
    if offDiag is None:
        offDiag=np.array([0]*int(numTraits*(numTraits-1)/2))
        
    np.savetxt('offDiag',offDiag,delimiter='\t')
    
    t1=time.time()
    
    b_time=bufCreate('time',[numCores])
    b_res=bufCreate('res',[numSnps])
    pids=[]
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(gbjHelp,func,core,z[snpRange],snpRange,b_time,b_res)]
        
    for pid in pids:
        os.waitpid(0, 0)
         
    log('{} : {} snps, {} min/snp'.format(func,numSnps,((t1-t0)+np.sum(bufClose(b_time)))/(60*numSnps)))
    
    return(bufClose(b_res))

def gbjHelp(func,core,z,snpRange,b_time,b_res):  
    t0=time.time()
    np.savetxt(func+'-'+str(core),z,delimiter='\t')
    subprocess.call(['Rscript','../source/R/gbj.R',func+'-'+str(core),func])
    b_res[0][snpRange]=np.loadtxt(func+'-'+str(core),delimiter='\t')
    b_res[1].flush()
    t1=time.time()
    
    b_time[0][core]=(t1-t0)
    b_time[1].flush()
    
    return()
