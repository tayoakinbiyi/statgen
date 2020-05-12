import pandas as pd
import numpy as np
import subprocess
import pdb
import os
from limix.model.lmm import LMM
from ELL.util import *
from multiprocessing import cpu_count
from utility import *
import warnings
import time

def runLimix(Y,QS,M,snps,etaMax=0.99):
    t0=time.time()
    
    numSnps=snps.shape[1]
    numTraits=Y.shape[1]
           
    wald=np.ones([numSnps,numTraits])    
    eta=np.ones([numTraits])
    reml=np.ones([1])
    fail=False

    warnings.simplefilter("error")
    
    for trait in range(Y.shape[1]):
        print('{} : {} of {}'.format('runLimix',trait,Y.shape[1]),flush=True)
        lmm = LMM(Y[:,trait], M, QS, restricted=False)
        lmm.fit(verbose=False)
        delta=lmm.v0/(lmm.v0+lmm.v1)
        if delta>etaMax:
            lmm = LMM(Y[:,trait], M, QS, restricted=True)
            lmm.fit(verbose=False)
            reml+=1
            delta=lmm.v0/(lmm.v0+lmm.v1)
            
            if delta>etaMax:
                fail=True                
                continue
        
        ret=lmm.get_fast_scanner().fast_scan(snps,False)
        
        wald[:,trait]=ret['effsizes1']/ret['effsizes1_se']        
        eta[trait]=delta
        
    if fail:
        print('failure even reml couldn\'t fit')
        return(wald,eta)

    warnings.simplefilter("default")
    log('runLimix reml Count {}'.format(reml))
    
    t1=time.time()
    log('{} : {} snps, {} traits, {} min'.format('limix',numSnps,numTraits,(t1-t0)/(60)))    
    
    return(wald,eta)

