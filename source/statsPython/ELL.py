import numpy as np
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

def ELL(psi,numCores,wald):
    t0=time.time()
    
    startPsiRowPerD=np.where(psi['eta']==0)[0]
    endPsiRowPerD=np.where(psi['eta']==1)[0]
        
    calD=len(startPsiRowPerD)
    Reps,numTraits=wald.shape

    b_pi=bufCreate('pi',[Reps,calD])
    b_pi[0][:]=np.sort(2*norm.sf(np.abs(wald)))[:,0:calD]
        
    b_time=bufCreate('time',[numCores])

    t1=time.time()
            
    pids=[]
    for core in range(numCores):
        dRange=np.arange(core*int(np.ceil(calD/numCores)),min(calD,(core+1)*int(np.ceil(calD/numCores))))
        if len(dRange)==0:
            continue
        #scoreHelp(psi,startPsiRowPerD,endPsiRowPerD,dRange,b_pi,b_time,core)   
           
        pids+=[remote(scoreHelp,psi,startPsiRowPerD,endPsiRowPerD,dRange,b_pi,b_time,core)]
    
    for pid in pids:
        os.waitpid(0, 0)
        
    t2=time.time()
    
    zeroPerD=np.where(np.sum(b_pi[0]==0,axis=0)>0)[0]
    zeroPerRep=np.where(np.sum(b_pi[0]==0,axis=1)>0)[0]

    onePerD=np.where(np.sum(b_pi[0]==1,axis=0)>0)[0]
    onePerRep=np.where(np.sum(b_pi[0]==1,axis=1)>0)[0]
    
    if len(zeroPerD)+len(onePerD)>0:
        log('{} under\n{}'.format(len(zeroPerD),zeroPerD))
        log('{} over\n{}'.format(len(onePerD),onePerD))
    
    score=np.min(bufClose(b_pi),axis=1)
    
    t3=time.time()

    return(score,((t1-t0)+(t3-t2)+np.sum(bufClose(b_time)))/60)

def scoreHelp(psi,startPsiRowPerD,endPsiRowPerD,dRange,b_pi,b_time,core):   
    t0=time.time()
    
    _,calD=b_pi[0].shape
     
    for d in dRange:
        t_psi=psi[startPsiRowPerD[d]:endPsiRowPerD[d]+1]
        arg=np.argsort(b_pi[0][:,d])
        loc=np.searchsorted(t_psi['lam'],b_pi[0][arg,d],side='left')
        b_pi[0][arg,d]=t_psi['eta'][loc]
    
    t1=time.time()
    
    b_time[0][core]=(t1-t0)
    b_time[1].flush()
    
    return()
