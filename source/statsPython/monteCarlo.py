import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *

def monteCarlo(func,wald,vZ,refReps,maxRefReps,numCores,name):
    memory('monteCarlo-'+name)
    
    test,mins=func(wald,numCores)
    log('{} : {} min'.format('mc score-'+name,mins))
    
    ref,mins=genRef(func,refReps,maxRefReps,vZ,numCores)
    log('{} : {} min'.format('mc genRef-'+name,mins))

    pval,mins=mcPVal(test,ref)
    log('{} : {} min'.format('mc pval-'+name,mins))
    
    memory('monteCarlo-'+name)
    
    return(pval)

def genRef(func,refReps,maxRefReps,vZ,numCores):    
    t0=time.time()
    memory('genRef')
    
    L=makeL(vZ)
    ref=np.ones(refReps)
    numTraits=vZ.shape[1]
    
    t1=time.time()
    
    mins=0
    length=0
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        repRange=np.arange(block*maxRefReps,min(refReps,(block+1)*maxRefReps)).astype(int)
        ans,t_mins=func(np.matmul(norm.rvs(size=[len(repRange),numTraits]),L.T),numCores)
        ref[repRange]=ans
        mins+=t_mins
        
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

