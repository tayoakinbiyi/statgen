import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from python.myStats import *
from python.ghc import *
from python.ggnull import *
from python.gbj import *

def fitMCMCStats(z,N,parms):
    cpus=parms['cpus']

    stat=pd.DataFrame()
        
    statELL=ELL(z,parms) # load ggNullDat from parms
    stat.insert(stat.shape[1],'ELL',statELL)
    print('ELL',psutil.virtual_memory().percent)

    statCPMA=cpma(z,parms)
    stat.insert(stat.shape[1],'cpma',statCPMA)
    print('cpma',psutil.virtual_memory().percent)

    Reps,d=z.shape
        
    futures=[]
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(int(np.ceil(Reps/np.ceil(Reps/cpus)))):
            futures.append(executor.submit(fitMCMCStatsHelp,z[i*int(np.ceil(Reps/cpus)):min((i+1)*int(np.ceil(Reps/cpus)),Reps)],N))
    
    power=pd.DataFrame(dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        result=f.result()
        power=power.append(result)
        
    stat=pd.concat([stat,statS],axis=1)
    print('myStats',psutil.virtual_memory().percent)
    
    # no return just write it to DB
    return(stat)

def fitMCMCStatsHelp(z,N):    
    Reps,d=z.shape

    p_val=2*norm.sf(z)

    Fn=(np.array([range(d)]*Reps)+1)/N
    Fn0=Fn[0]

    minP=-p_val[:,0]   
    hc=[np.sqrt(N)*np.max(((Fn0 -p)/np.sqrt(p*(1-p)))[p>=1/N]) for p in p_val]
    hcFull=np.sqrt(N)*np.max((Fn-p_val)/np.sqrt(p_val*(1-p_val)),axis=1)
    bj=np.max(N*D(Fn,p_val),axis=1)
    gnull=np.max(-beta.cdf(p_val,np.array([range(1,d+1)]*Reps),N-np.array([range(1,d+1)]*Reps)),axis=1)
    score=np.sum(z**2,axis=1)
    
    #pd.DataFrame({'Type':'hcFull','Value':hcFull}),
    power=pd.concat([
        pd.DataFrame({'Type':'hc','Value':hc}),
        pd.DataFrame({'Type':'minP','Value':minP}),
        pd.DataFrame({'Type':'bj','Value':bj}),
        pd.DataFrame({'Type':'gnull','Value':gnull}),
        pd.DataFrame({'Type':'score','Value':score})
    ],axis=0).reset_index(drop=True)

    return(power)