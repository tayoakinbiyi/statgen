import numpy as np
import pdb
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
import time

from ELL.util import *
from multiprocessing import cpu_count
from limix.model.lmm import LMM
from numpy_sugar.linalg import economic_qs

def runH1(mu,eps,z,Y,QS,M,snps):     
    numSnps=snps.shape[1]
    numTraits=Y.shape[1]
                            
    b_waldStat=bufCreate('waldStat',[numSnps,numTraits])
    b_waldStat[0][:]=z
    
    pids=[]
    numCores=cpu_count()
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(runH1Help,mu,eps,Y,QS,M,snps,b_waldStat,snpRange)]
        
    for pid in pids:
        os.waitpid(0, 0)
        
    return(bufClose(b_waldStat))

def runH1Help(mu,eps,Y,QS,M,snps,b_waldStat,snpRange):
    numTraits=Y.shape[1]
    m=M.shape[1]
    
    for snp in snpRange:
        traitRange=np.random.choice(range(numTraits),eps)
        maf=np.mean(snps[:,snp])/2
        Mhat=np.concatenate([M,snps[:,snp:snp+1]],axis=1)
        count=1
        for trait in traitRange:
            print('{} : {} , {} of {}'.format('runH1',snp,count,len(traitRange)),flush=True)

            y=Y[:,trait]+(mu/np.sqrt(2*eps*maf*(1-maf)))*np.random.choice([-1,1],size=1)*snps[:,snp]
            lmm = LMM(y, Mhat, QS, restricted=False)
            lmm.fit(verbose=False)
            b_waldStat[0][snp,trait]=lmm.beta[m]/np.sqrt(lmm.beta_covariance[m,m])
            count+=1
    
    return()
