import numpy as np
import pdb
import os

import time
from utility import *
from statsPython.ELL import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import numpy as np
from scipy.stats import norm

def markov(psi,vZ,numCores,wald,dfName):
    t0=time.time()
    memory('markov')
    
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
        
    reps,D=wald.shape
    
    stat,mins=ELL(psi,numCores,wald)
    offDiagVec=ro.FloatVector(tuple(offDiag*0))
    
    startPsiRowPerD=np.where(psi['eta']==0)[0]
    endPsiRowPerD=np.where(psi['eta']==1)[0]

    calD=len(startPsiRowPerD)
            
    b_z=bufCreate('z',[reps,calD])
    b_markov=bufCreate('markov',[reps])
    b_time=bufCreate('time',[numCores])
    
    t1=time.time()

    pids=[]
    for core in range(numCores):
        dRange=np.arange(core*int(np.ceil(calD/numCores)),min(calD,(core+1)*int(np.ceil(calD/numCores))))

        if len(dRange)==0:
            continue
        #reverseScore(dRange,stat,psi,startPsiRowPerD,endPsiRowPerD,b_z)
        pids+=[remote(reverseScore,core,dRange,stat,psi,startPsiRowPerD,endPsiRowPerD,b_z,b_time)]

    for pid in pids:
        os.waitpid(0, 0)
    
    pids=[]
    for core in range(numCores):
        repRange=np.arange(core*int(np.ceil(reps/numCores)),min(reps,(core+1)*int(np.ceil(reps/numCores))))

        if len(repRange)==0:
            continue
        #markovHelp(repRange,b_z,b_markov,offDiagVec,calD,D)
        pids+=[remote(markovHelp,core,repRange,b_z,b_markov,b_time,offDiagVec,calD,D)]

    for pid in pids:
        os.waitpid(0, 0)

    t2=time.time()

    pvals=bufClose(b_markov)

    memory('markov')
    
    t3=time.time()
    
    log('beginning score for {} {}, {} min'.format('ELL-markov',dfName,((t1-t0)+(t3-t2)+np.sum(bufClose(b_time)))/60))
        
    return(pvals.reshape(-1,1))

def markovHelp(core,repRange,b_z,b_markov,b_time,offDiagVec,calD,D):
    t0=time.time()
    gbj=importr('GBJ')

    row=np.ones([D])
    for rep in repRange:
        row[0:calD]=b_z[0][rep]
        row[calD:]=row[calD-1]
        bounds=ro.FloatVector(row[::-1])
        b_markov[0][rep]=gbj.ebb_crossprob_cor_R(d=D, bounds=bounds, correlations=offDiagVec)[0]
        b_markov[1].flush()
    
    t1=time.time()
    
    b_time[0][core]+=(t1-t0)
    b_time[1].flush()
    
    return()
    
def reverseScore(core,dRange,stats,psi,startPsiRowPerD,endPsiRowPerD,b_z,b_time):
    t0=time.time()
    
    for d in dRange:
        t_psi=psi[startPsiRowPerD[d]:endPsiRowPerD[d]+1]
        arg=np.argsort(stats)
        loc=np.maximum(np.searchsorted(t_psi['eta'],stats[arg],side='left'),1)
        b_z[0][arg,d]=-norm.ppf(t_psi['lam'][loc]/2)
        b_z[1].flush()
    
    t1=time.time()
    
    b_time[0][core]=(t1-t0)
    
    return()