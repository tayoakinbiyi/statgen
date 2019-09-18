import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from ail.statsPython.ELL import *
from ail.opPython.DB import *

def MCMCStats(z,parms):
    IIDReps=parms['IIDReps']
    
    if not DBIsFile(name+'sim','IIDStats',parms):
        genIIDStats(parms)
    else:
        iidStats=DBRead(name+'sim/IIDStats',parms,toPickle=True)
    
    stats=fitMCMCStats(z,parms)
    
    Types=stats.Type.drop_duplicates().values.flatten().tolist()
    
    pvals=pd.DataFrame()
    for Type in Types:
        pval=iidStats.loc[iidStats['Type']==Type,'Value'].searchsorted(stats.loc[stats['Type']==Type,'Value'])/float(IIDReps)
        pvals=pvals.append(pd.DataFrame({'Type':Type,'Value':pval}))
        
    return(pvals)

def fitMCMCStats(z,parms):
    cpus=parms['cpus']
    ellDSet=parms['ellDSet']

    N=DBRead(name+'process/N',parms,toPickle=False)
    stats=pd.DataFrame()
        
    for dParm in ellDSet:
        stats=stats.append(ELL(z,dParm,parms))
    
    stats=stats.append(cpma(z,parms))
    
    Reps,d=z.shape
        
    futures=[]
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(int(np.ceil(Reps/np.ceil(Reps/cpus)))):
            futures.append(executor.submit(fitMCMCStatsHelp,z[i*int(np.ceil(Reps/cpus)):min((i+1)*int(np.ceil(Reps/cpus)),Reps)],N))
    
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        result=f.result()
        stats=stats.append(result)
    
    return(stats.sort_values(by=['Type','Value']))

def fitMCMCStatsHelp(z,N):    
    Reps,d=z.shape

    p_val=2*norm.sf(z)

    Fn=(np.array([range(d)]*Reps)+1)/N
    Fn0=Fn[0]

    hc=[np.sqrt(N)*np.max(((Fn0 -p)/np.sqrt(p*(1-p)))[p>=1/N]) for p in p_val]
    bj=np.max(N*D(Fn,p_val),axis=1)
    gnull=np.max(-beta.cdf(p_val,np.array([range(1,d+1)]*Reps),N-np.array([range(1,d+1)]*Reps)),axis=1)
    score=np.sum(z**2,axis=1)
    
    stats=pd.concat([
        pd.DataFrame({'Type':'hc','Value':hc}),
        pd.DataFrame({'Type':'bj','Value':bj}),
        pd.DataFrame({'Type':'gnull','Value':gnull}),
        pd.DataFrame({'Type':'score','Value':score})
    ],axis=0).reset_index(drop=True)

    return(stats)

    stats=pd.DataFrame(dtype='float32')

def genIIDStats(parms):    
    IIDReps=parms['IIDReps']
    name=parms['name']
    
    LZCorr=DBRead(name+'process/LZCorr-all',parms,toPickle=True)
    N=DBRead(name+'process/N',parms,toPickle=False)

    z=np.matmul(norm.rvs(size=[IIDReps,N]),LZCorr.T)
    z=-np.sort(-np.abs(z))[:,0:int(N/2)]
        
    stats=MCMCStats(z,parms).sort_values(by=['Type','Value'])
    
    DBWrite(stats,name+'sim/IIDStats',parms,toPickle=True)
    
    return()