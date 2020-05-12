from opPython.DB import *
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pdb
from ELL.util import *
from multiprocessing import cpu_count

def gbj(func,z,name,offDiag=None): 
    numSnps,numTraits=z.shape
                 
    if offDiag is None:
        offDiag=np.array([0]*int(numTraits*(numTraits-1)/2))
        
    b_pval=bufCreate('pval',[numSnps])
    b_pval[0][:]=0
    
    pids=[]
    numCores=cpu_count()
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(gbjHelp,func,b_pval,z,snpRange,offDiag)]
        
    for pid in pids:
        os.waitpid(0, 0)
    
    pval=pd.DataFrame(bufClose(b_pval),columns=[name])
    
    return(pval)

def gbjHelp(func,b_pval,z,snpRange,offDiag)
    pval=np.arange(Reps).astype(float)    
    offDiagVec=ro.FloatVector(tuple(offDiag))
    
    count=0
    for snp in snpRange:
        print('{} of {}'.format(count, len(snpRange)))
        vec=ro.FloatVector(tuple(z[snp]))
        
        b_pval[0][snp]=func(test_stats=vec,  pairwise_cors=offDiagVec)[1][0]
    
    return()
