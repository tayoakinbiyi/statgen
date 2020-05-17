import numpy as np
import pdb
from scipy.stats import norm
import pandas as pd
import os

from ELL.util import *
from utility import *
import numpy as np
import pdb
import os
import time

import time

def f(lamEllByK,ellGrid,wald,numCores):
    t0=time.time()
    memory('score')
    
    numRows=len(wald)
    d=lamEllByK.shape[1]

    wald=np.sort(2*norm.sf(np.abs(wald)))[:,0:d]
    
    b_check=bufCreate('check',[3,d])
    
    b_time=bufCreate('time',[numCores])

    t1=time.time()
            
    pids=[]
    b_score={}
    for core in range(numCores):
        kRange=np.arange(core*int(np.ceil(d/numCores)),min(d,(core+1)*int(np.ceil(d/numCores))))
        if len(kRange)==0:
            continue
        b_score[core]=bufCreate('pvals-'+str(core),[wald.shape[0],len(kRange)])
        b_score[core][0][:]=wald[:,kRange]
        
        pids+=[remote(scoreHelp,kRange,b_score[core],lamEllByK,b_check,b_time,core)]

    for pid in pids:
        os.waitpid(0, 0)
        
    t2=time.time()
    check=b_check[0][:,np.min(b_check[0][1:],axis=0)>0]
    print(pd.DataFrame(check[1:],columns=check[0],index=['below','above']),flush=True)
    
    wald=np.concatenate([b_score[core][0] for core in range(len(b_score))],axis=1)
    
    ellStats=np.min(wald[:,0:d],axis=1).flatten().astype(int)
    
    for core in b_score:
        bufClose(b_score[core])
    bufClose(b_check)    

    t3=time.time()

    memory('score')
        
    return(ellGrid[ellStats],((t1-t0)+np.sum(bufClose(b_time))+(t3-t2))/60)

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
