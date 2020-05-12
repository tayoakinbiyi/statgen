import pandas as pd
import numpy as np
import pdb
import subprocess
from ail.opPython.DB import *
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

def iidLocalLevels(N,alpha):       
    iidLocalLevelsString='\n'.join(np.loadtxt('../ext/iidLocalLevels',delimiter='\t'))
    iidLocalLevels=SignatureTranslatedAnonymousPackage(iidLocalLevelsString,'iidLocalLevels')    

    eta0=0
    eta1=alpha
    f0=0
    f1=iidLocalLevels.iidLocalLevels(ro.FloatVector(tuple(beta.ppf(eta1,np.arange(1,N+1),np.arange(1,N+1)[::-1]))))
    while eta1-eta0>1e-4:
        newEta=(eta0+eta1)/2
        fnew=iidLocalLevels.iidLocalLevels(ro.FloatVector(tuple(beta.ppf(newEta,np.arange(1,N+1),np.arange(1,N+1)[::-1]))))
        if fnew<alpha:
            eta0=newEta
        else:
            eta1=newEta
    
    return(eta0)
        