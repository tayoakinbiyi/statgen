import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *

def monteCarlo(self,ellStat,refReps,maxRefReps):
    self.genRef(refReps,maxRefReps)
    ans=self.monteCarloPval(ellStat)
    
    del self.mcRef
    
    return(ans)

def genRef(self,refReps,maxRefReps):    
    t0=time.time()
    memory('genRef')
    
    ref=[]
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        reps=int(min(refReps,(block+1)*int(np.ceil(refReps/maxRefReps)))-block*int(np.ceil(refReps/maxRefReps)))
        
        ref+=[self.score(-np.sort(-np.abs(np.matmul(norm.rvs(size=[reps,self.N]),self.L.T)),
            axis=0)[:,0:self.d],verbose=False)]
    
    self.mcRef=np.concatenate(ref)
    
    memory('genRef')
    t1=time.time()
    
    log('{} : {} min'.format('genRef',(t1-t0)/(60)))

    return()

def monteCarloPval(self,ellStat):
    t0=time.time()
    memory('monteCarlo')
    
    refReps=self.mcRef.shape[0]

    mc=np.zeros(ellStat.shape)

    sortOrd=np.argsort(ellStat,axis=0)
    mc[sortOrd]=(1+np.searchsorted(np.sort(self.mcRef),ellStat[sortOrd]))/(refReps+1)
        
    memory('monteCarlo')
    t1=time.time()
    
    log('{} : {} min'.format('monteCarlo',(t1-t0)/(60)))

    return(mc)

