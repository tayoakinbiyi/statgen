import pandas as pd
import numpy as np
import pdb
import subprocess
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from ELL.util import *
from scipy.stats import beta

def iidLocalLevels(stats,repRange,N,d):  
    iidLocalLevelsString='\n'.join(np.loadtxt('../ext/iidLocalLevels',delimiter='\t',dtype=str))
    f=SignatureTranslatedAnonymousPackage(iidLocalLevelsString,'iidLocalLevels')    
    
    kRan1=np.arange(1,d+1)
    kRan2=np.arange(N,N-d,-1)
    h=np.ones(N)
    
    count=1
    for rep in repRange:        
        print('{} of {}'.format(count,len(repRange)))
        h[0:d]=beta.ppf(stats[0][rep],kRan1,kRan2)
        h[d:]=h[d-1]+np.arange(1,N-d+1)*np.exp(-30)
        stats[0][rep]=f.iidLocalLevels(ro.FloatVector(tuple(h)))[0]
        count+=1
    
    return()

def exactHelp(b_score,kRange,N):
    Reps,numK=b_score[0].shape
    for kInd in np.arange(numK):
        k=kRange[kInd]
        b_score[0][:,kInd]=beta.cdf(b_score[0][:,kInd],k+1,N-k)
        
    return()