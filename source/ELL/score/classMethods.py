import numpy as np
import pdb
import ray
from scipy.stats import norm
import ray
import pandas as pd
import os

from ELL.score.remotes import *
from ELL.util import memory

def score(self,testStats):
    memory('score')

    numCores=self.numCores  
    dList=self.dList
    maxD=dList[-1]
    
    r_lamEllByK=ray.put(self.lamEllByK)
    r_pvals=ray.put(np.sort(2*norm.sf(np.abs(testStats))))
    r_check=ray.put(np.zeros([3,maxD]),weakref=True)
    
    print('score ({}): r_lamEllByK {}, get(r_lamEllByK) {}'.format(os.getpid(), id(r_lamEllByK),id(ray.get(r_lamEllByK))))
    
    objectIds=[]
    for core in range(numCores):
        kRange=np.arange(core*int(np.ceil(maxD/numCores)),min(maxD,(core+1)*int(np.ceil(maxD/numCores))))
        if len(kRange)==0:
            continue
            
        objectIds+=[scoreHelp.remote(ray.put(kRange),r_pvals,r_lamEllByK,r_check)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))
        
    check=ray.get(r_check)
    check=check[:,np.min(check[1:],axis=0)>0]
    if len(check)>0:
        print(pd.DataFrame(check[1:],columns=check[0],index=['below','above']),flush=True)
        
    ellStats=np.zeros([len(testStats),len(dList)],dtype=int)
    for dInd in range(len(dList)):
        ellStats[:,dInd]=np.min(ray.get(r_pvals)[:,0:dList[dInd]],axis=1)

    ellGrid=self.ellGrid
    
    memory('score')

    return(ellGrid[ellStats])



