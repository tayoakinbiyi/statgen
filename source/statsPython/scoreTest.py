import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *

def scoreTest(wald,refReps,maxRefReps,vZ):
    ans=monteCarloScoreTestPval(scoreTestScore(wald),genScoreTestRef(refReps,maxRefReps,vZ))
    
    return(ans)

def scoreTestScore(wald):
    t0=time.time()
    memory('scoreTestScore')
    
    ans=np.sum(wald**2,axis=1)
    
    memory('scoreTestScore')
    t1=time.time()
    
    log('{} : {} snps {} min/snp'.format('scoreTestScore',len(ans),(t1-t0)/(60)))
    
    return(ans)


def genScoreTestRef(refReps,maxRefReps,vZ):    
    t0=time.time()
    memory('genScoreTestRef')
    
    L=makeL(vZ)
    
    ref=[]
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        reps=int(min(refReps,(block+1)*int(np.ceil(refReps/maxRefReps)))-block*int(np.ceil(refReps/maxRefReps)))
        
        ref+=[scoreTestScore(-np.sort(-np.abs(np.matmul(norm.rvs(size=[reps,self.N]),self.L.T)),
            axis=0)[:,0:self.d],verbose=False)]
    
    ans=np.concatenate(ref)
    
    memory('genScoreTestRef')
    t1=time.time()
    
    log('{} : {} min'.format('genScoreTestRef',(t1-t0)/(60)))

    return(ans)

def monteCarloScoreTestPval(test,ref):
    t0=time.time()
    memory('monteCarloScoreTestPval')
    
    mc=np.zeros(test.shape)
    refReps=ref.shape[0]

    sortOrd=np.argsort(test,axis=0)
    mc[sortOrd]=1-(1+np.searchsorted(np.sort(ref),test[sortOrd]))/(refReps+1)
        
    memory('monteCarloScoreTestPval')
    t1=time.time()
    
    log('{} : {} min'.format('monteCarloScoreTestPval',(t1-t0)/(60)))

    return(mc)

