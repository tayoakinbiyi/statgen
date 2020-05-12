import numpy as np
from scipy.stats import norm
import pdb
from ELL.exact.remotes import *
from ELL.util import *

def exact(self,testStats):      
    memory('exact')

    numCores=self.numCores
    N=self.N    
    d=self.d
    reps=testStats.shape[0]

    testStats=np.sort(2*norm.sf(np.abs(testStats)))[:,0:d]
    
    pids=[]
    b_score={}
    for core in range(numCores):
        kRange=np.arange(core*int(np.ceil(d/numCores)),min(d,(core+1)*int(np.ceil(d/numCores))))
        if len(kRange)==0:
            continue
        b_score[core]=bufCreate('pvals-'+str(core),[testStats.shape[0],len(kRange)])
        b_score[core][0][:]=testStats[:,kRange]
            
        pids+=[remote(exactHelp,b_score[core],kRange,self.N)]
        
    for pid in pids:
        os.waitpid(0, 0)
    
    ellStats=bufCreate('ellStats',[testStats.shape[0]])
    ellStats[0][:]=np.min(np.concatenate([b_score[core][0] for core in range(len(b_score))],axis=1),axis=1)
    
    for core in b_score:
        bufClose(b_score[core])
    
    pids=[]
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))
        if len(repRange)==0:
            continue
        
        pids+=[remote(iidLocalLevels,ellStats,repRange,N,d)]
        
    for pid in pids:
        os.waitpid(0, 0)

    memory('exact')

    return(bufClose(ellStats))

