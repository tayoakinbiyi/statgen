import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *
from multiprocessing import cpu_count
from ELL.util import *

def storeyQ(wald,refReps,maxRefReps,vZ,d,numCores=cpu_count()):
    memory('storeyQ')
    
    pi=2*norm.sf(np.abs(wald))
    ans1,delta1=storeyQScore(pi,d,numCores)
    log('{} : {} min'.format('storeyQScore',delta1))

    ans2,delta2=genStoreyQRef(refReps,maxRefReps,vZ,d,numCores)
    log('{} : {} min'.format('genStoreyQRef',delta2))
    
    ans,delta3=monteCarloStoreyQPval(ans1,ans2)
    log('{} : {} min'.format('monteCarloStoreyQPval',delta3))
    
    memory('storeyQ')

    return(ans)

def storeyQScore(pi,d,numCores):
    t0=time.time()

    reps=pi.shape[0]
    
    pids=[]
    b_storeyQ=bufCreate('storeyQ',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(storeyQScoreHelp,pi[repRange],d,b_storeyQ,repRange)]

    for pid in pids:
        os.waitpid(0, 0)
        
    ans=bufClose(b_storeyQ)
    
    t1=time.time()
        
    return(ans,numCores*(t1-t0)/(60))

def storeyQScoreHelp(pi,d,b_storeyQ,repRange):
    b_storeyQ[0][repRange]=np.min(np.sort(pi*np.arange(1,pi.shape[1]+1).reshape(1,-1))[:,0:d],axis=1)
    b_storeyQ[1].flush()
    
    return()

def genStoreyQRef(refReps,maxRefReps,vZ,d,numCores):    
    L=makeL(vZ)
    numTraits=vZ.shape[1]
    
    ref=[]
    delta=0
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        reps=int(min(refReps,(block+1)*int(np.ceil(refReps/maxRefReps)))-block*int(np.ceil(refReps/maxRefReps)))
    
        pi=2*norm.sf(np.abs(np.matmul(norm.rvs(size=[reps,numTraits]),L.T)))
        tmp,t_delta=storeyQScore(pi,d,numCores)
        ref+=[tmp]
        delta+=t_delta
    
    ans=np.concatenate(ref)
    
    return(ans,delta)

def monteCarloStoreyQPval(test,ref):
    t0=time.time()
    
    mc=np.zeros(test.shape)
    refReps=ref.shape[0]

    sortOrd=np.argsort(test,axis=0)
    mc[sortOrd]=(1+np.searchsorted(np.sort(ref),test[sortOrd]))/(refReps+2)
        
    t1=time.time()
    
    return(mc,(t1-t0)/(60))

