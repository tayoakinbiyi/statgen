import pandas as pd
import numpy as np
from scipy.stats import norm, beta
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from myMath import *

def myStats(z):
    Reps,N=z.shape
    d=int(N/2)
    
    z=-np.sort(-np.abs(z))[:,0:d]
    
    M=multiprocessing.cpu_count()

    with ProcessPoolExecutor() as executor: 
        results=executor.map(myStatsHelp, [z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)].tolist()
            for i in range(int(np.ceil(Reps/np.ceil(Reps/M))))])
    
    power=pd.DataFrame()
    for result in results:   
        power=power.append(result)

    return(power)

def myStatsHelp(dat):
    z=np.array(dat)
    
    Reps,d=z.shape
    N=int(d*2)

    p_val=2*norm.sf(z)

    Fn=(np.array([range(d)]*Reps)+1)/N
    Fn0=Fn[0]

    minP=-p_val[:,0]   
    hc=[np.sqrt(N)*np.max(((Fn0 -p)/np.sqrt(p*(1-p)))[p>=1/N]) for p in p_val]
    hcFull=np.sqrt(N)*np.max((Fn-p_val)/np.sqrt(p_val*(1-p_val)),axis=1)
    bj=np.max(N*D(Fn,p_val),axis=1)
    gnull=np.max(-beta.cdf(p_val,np.array([range(1,d+1)]*Reps),N-np.array([range(1,d+1)]*Reps)),axis=1)
    fdr=np.max(Fn/p_val,axis=1)
    score=np.sum(z**2,axis=1)
    
    power=pd.concat([
        pd.DataFrame({'Type':'hc','Value':hc}),
        pd.DataFrame({'Type':'hcFull','Value':hcFull}),
        pd.DataFrame({'Type':'minP','Value':minP}),
        pd.DataFrame({'Type':'bj','Value':bj}),
        pd.DataFrame({'Type':'gnull','Value':gnull}),
        pd.DataFrame({'Type':'fdr','Value':fdr}),
        pd.DataFrame({'Type':'score','Value':score})
    ],axis=0).reset_index(drop=True)

    return(power)

