import numpy as np
import pdb
from scipy.stats import norm, beta
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
    testStats=np.sort(2*norm.sf(np.abs(testStats)))[:,0:maxD]
    
    b_check=bufCreate('check',[3,maxD])
            
    pids=[]
    b_score={}
    for core in range(numCores):
        kRange=np.arange(core*int(np.ceil(maxD/numCores)),min(maxD,(core+1)*int(np.ceil(maxD/numCores))))
        if len(kRange)==0:
            continue
        b_score[core]=bufCreate('pvals-'+str(core),[testStats.shape[0],len(kRange)])
        b_score[core][0][:]=testStats[:,kRange]
        
        pids+=[remote(scoreHelp,kRange,b_score[core],lamEllByK,b_check)]

    for pid in pids:
        os.waitpid(0, 0)
        
    check=b_check[0][:,np.min(b_check[0][1:],axis=0)>0]
    if len(check)>0:
        print(pd.DataFrame(check[1:],columns=check[0],index=['below','above']),flush=True)
    
    testStats=np.concatenate([b_score[core][0] for core in range(len(b_score))],axis=1)
    ellStats=np.zeros([len(testStats),len(dList)],dtype=int)
    
    for dInd in range(len(dList)):
        ellStats[:,dInd]=np.min(testStats[:,0:dList[dInd]],axis=1)
    
    ellGrid=self.ellGrid

    for core in b_score:
        bufClose(b_score[core])
    bufClose(b_check)    
    
    memory('score')

    return(ellGrid[ellStats])



