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
    
    pids=[]
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(gbjHelp,func,core,z[snpRange])]
        
    for pid in pids:
        os.waitpid(0, 0)
        
    z=[]
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        z+=[np.loadtxt(func+'-'+str(core),delimiter='\t').reshape(-1,1)]
    z=np.concatenate(z,axis=0).flatten()
   
    t1=time.time()
    
    log('{} : {} snps, {} min/snp'.format(func,numSnps,numCores*(t1-t0)/(60*numSnps)))
    
    return(z)

def gbjHelp(func,core,z):     
    np.savetxt(func+'-'+str(core),z,delimiter='\t')
    subprocess.call(['Rscript','../source/R/gbj.R',func+'-'+str(core),func])
    
    return()
