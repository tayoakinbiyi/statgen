import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *
from multiprocessing import cpu_count
from ELL.util import *

def scoreTest(wald,refReps,maxRefReps,vZ,numCores=cpu_count()):
    memory('scoreTest')

    ans1,delta1=scoreTestScore(wald,numCores)
    log('{} : {} min'.format('scoreTestScore',delta1))

    ans2,delta2=genScoreTestRef(refReps,maxRefReps,vZ,numCores)
    log('{} : {} min'.format('genScoreTestRef',delta2))

    ans,delta3=monteCarloScoreTestPval(ans1,ans2)
    log('{} : {} min'.format('genScoreTestRef',delta3))
    
    memory('scoreTest')

    return(ans)

def scoreTestScore(wald,numCores):
    t0=time.time()
    
    reps=wald.shape[0]
    
    pids=[]
    b_score=bufCreate('score',[reps])
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        
        pids+=[remote(scoreTestScoreHelp,wald[repRange],b_score,repRange)]

    for pid in pids:
        os.waitpid(0, 0)
        
    ans=bufClose(b_score)
    
    t1=time.time()
    
    return(ans,numCores*(t1-t0)/60)

def scoreTestScoreHelp(wald,b_score,repRange):
    b_score[0][repRange]=np.sum(wald**2,axis=1)
    return()

def genScoreTestRef(refReps,maxRefReps,vZ,numCores):    
    L=makeL(vZ)
    numTraits=vZ.shape[1]
    
    ref=[]
    delta=0
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        reps=int(min(refReps,(block+1)*int(np.ceil(refReps/maxRefReps)))-block*int(np.ceil(refReps/maxRefReps)))
        tmp,t_delta=scoreTestScore(np.matmul(norm.rvs(size=[reps,numTraits]),L.T),numCores)
        ref+=[tmp]
        delta+=t_delta
    
    ans=np.concatenate(ref)

    return(ans,delta)

def monteCarloScoreTestPval(test,ref):
    t0=time.time()
    
    mc=np.zeros(test.shape)
    refReps=ref.shape[0]

    sortOrd=np.argsort(test,axis=0)
    mc[sortOrd]=1-(1+np.searchsorted(np.sort(ref),test[sortOrd]))/(refReps+2)
        
    t1=time.time()
    
    return(mc,(t1-t0)/60)

