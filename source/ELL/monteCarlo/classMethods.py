import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *

def monteCarlo(func,wald,vZ,refReps,maxRefReps,numCores,name):
    memory('monteCarlo-'+name)

    test,mins=func(wald,numCores)
    log('{} : {} min'.format('mc score-'name,mins))

    ref,mins=genRef(func,refReps,maxRefReps,vZ,numCores)
    log('{} : {} min'.format('mc genRef-'+name,mins))

    pval,mins=mcPVal(test,ref)
    log('{} : {} min'.format('mc pval-'+name,mins))
    
    memory('monteCarlo-'+name)
    
    return(pval)

def genRef(func,refReps,maxRefReps,vZ):    
    t0=time.time()
    memory('genRef')
    
    L=makeL(vZ)
    mc=np.ones(refReps)
    
    t1=time.time()
    
    ref=[]
    mins=0
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        repRange=np.arange(block*int(np.ceil(refReps/maxRefReps)),min(refReps,(block+1)*int(np.ceil(refReps/maxRefReps))))
        
        ans,t_mins=func(norm.rvs(size=[len(repRange),self.N]),L.T)
        ref[repRange]=ans
        length+=t_mins
        
    memory('genRef')
    
    return(ref,(t1-t0)/60+mins)

def mcPVal(test,ref):
    t0=time.time()
    memory('monteCarlo')
    
    refReps=len(ref)

    mc=np.zeros(len(test))

    sortOrd=np.argsort(test)
    mc[sortOrd]=(1+np.searchsorted(np.sort(ref),test[sortOrd]))/(refReps+1)
        
    memory('monteCarlo')
    t1=time.time()
    
    return(mc,(t1-t0)/60)

