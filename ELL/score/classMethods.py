import numpy as np
import pdb
import ray
from scipy.stats import norm
import ray

from ELL.score.remotes import *

def score(self,testStats):
    self.memory('score',start=True)

    numCores=self.numCores  
    maxD=self.dList[-1]
    
    r_lamEllByK=self.r_lamEllByK
    r_pvals=ray.put(np.sort(2*norm.sf(np.abs(testStats))))
    r_check=ray.put(np.zeros([3,maxD]))
    
    objectIds=[]
    for core in range(numCores):
        kRange=np.clip(np.arange(core*int(np.ceil(maxD/numCores)),(core+1)*int(np.ceil(maxD/numCores))),0,maxD-1)
        if len(kRange)==0:
            continue
            
        objectIds+=[scoreHelp.remote(ray.put(kRange),r_pvals,r_lamEllByK,r_check)]

    ready_ids, remaining_ids = ray.wait(objectIds, num_returns=len(objectIds))
        
    check=ray.get(r_check)
    print(check[:,np.min(check[1:],axis=0)>0],flush=True)
    
    self.scoreDone=True
    self.reps=len(testStats)
    
    self.ellStats=np.empty([self.reps,len(self.dList)])
    for dInd in range(len(self.dList)):
        self.ellStats[:,dInd]=ray.put(np.min(ray.get(r_pvals)[:,0:dList[dInd]],axis=1))

    ellGrid=ray.get(self.ellGrid)

    ans=np.concatenate([ellGrid[loc] for loc in self.ellStats.T],axis=1)
    
    self.memory('score')

    return(ans)



