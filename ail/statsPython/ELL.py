from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil

def ELL(zAll,dParam,parms):
    cpus=parms['cpus']
    Reps,N=zAll.shape
    
    ELLDat=DBRead(name+'sim/ELL-'+str(dParam),parms,toPickle=True)
    d=len(ELLDat)

    z={}
    for k in range(d):
        z[k]=zAll[:,k]
    del zAll
    
    futures=[]
    with ProcessPoolExecutor(cpus) as executor: 
        for k in range(d):
            futures.append(executor.submit(ELLHelp,z[k],ELLDat[k],k))
    
        ELLStat=[]
        for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
            ELLStat+=[f.result()]
    
    ELLStat=pd.DataFrame({'Type':'ELL','Value':-np.concatenate(ELLStat,axis=1).min(axis=1)})
    
    return(ELLStat)
    
def ELLHelp(z,ELLDat,k):   
    Reps=len(z)   
    p_vals=2*norm.sf(z)
    sortOrd=p_vals.argsort()
                 
    loc=ELLDat['binEdge'].searchsorted(p_vals[sortOrd])
    if max(loc)>=len(ELLDat):
        print(k,min(p_vals),max(p_vals))
        
    ELLStat=ELLDat['ELL'].iloc[np.minimum(loc,len(ELLDat)-1)].iloc[np.argsort(sortOrd)].values.flatten()
    
    return(ELLStat)
        