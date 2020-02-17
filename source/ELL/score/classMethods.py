import numpy as np
import pdb
import ray
from scipy.stats import norm
import ray
import pandas as pd
import os

from ELL.score.remotes import *
from ELL.util import *

def score(self,testStats):
    memory('score')

    numCores=self.numCores  
    dList=self.dList
    maxD=dList[-1]
    
    lamEllByK=self.lamEllByK
    b_pvals=bufCreate('pvals',[testStats.shape[0],maxD])
    b_pvals[0][:]=np.sort(2*norm.sf(np.abs(testStats)))[:,0:maxD]

    b_check=bufCreate('check',[3,maxD])
            
    pids=[]
    for core in range(numCores):
        kRange=np.arange(core*int(np.ceil(maxD/numCores)),min(maxD,(core+1)*int(np.ceil(maxD/numCores))))
        if len(kRange)==0:
            continue
        
        pids+=[remote(scoreHelp,kRange,b_pvals,lamEllByK,b_check)]

    for pid in pids:
        os.waitpid(0, 0)
        
    check=b_check[0][:,np.min(b_check[0][1:],axis=0)>0]
    if len(check)>0:
        print(pd.DataFrame(check[1:],columns=check[0],index=['below','above']),flush=True)
        
    ellStats=np.zeros([len(testStats),len(dList)],dtype=int)
    
    for dInd in range(len(dList)):
        ellStats[:,dInd]=np.min(b_pvals[0][:,0:dList[dInd]],axis=1)
    
    bufClose(b_pvals)
    bufClose(b_check)
    
    ellGrid=self.ellGrid
    
    memory('score')

    return(ellGrid[ellStats])



