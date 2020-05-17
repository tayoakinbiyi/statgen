import numpy as np
import pdb
import os
from ELL.F import *
import time

def scoreHelp(kRange,b_score,lamEllByK,b_check,b_time,core):   
    t0=time.time()
    for kInd in range(len(kRange)):
        k=kRange[kInd]
        b_check[0][:,k]=[k,np.mean(b_score[0][:,kInd]<lamEllByK[0,k]),np.mean(b_score[0][:,kInd]>lamEllByK[-1,k])]
        sortOrd=np.argsort(b_score[0][:,kInd])
        b_score[0][sortOrd,kInd]=np.clip(np.searchsorted(lamEllByK[:,k],b_score[0][sortOrd,kInd],side='left'),0,
            lamEllByK.shape[0]-1)   
        b_score[1].flush()
        
    t1=time.time()
    
    b_time[0][core]=(t1-t0)
    b_time[1].flush()
    
    return()
