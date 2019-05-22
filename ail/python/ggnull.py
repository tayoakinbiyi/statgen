from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil

def ggnull(zAll,sigName,ggnullDat):
    Reps,N=zAll.shape
    d=int(N/2)

    zAll=-np.sort(-np.abs(zAll))[:,0:d]
    z={}
    for k in range(d):
        z[k]=zAll[:,k]
    del zAll
    
    M=multiprocessing.cpu_count()
    
    futures=[]
    with ProcessPoolExecutor() as executor: 
        for k in range(d):
            futures.append(executor.submit(ggHelp,z[k],ggnullDat[k],k))
    
    ggnull=pd.DataFrame(dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        result=f.result()
        ggnull.insert(ggnull.shape[1],result[0],result[1])
    
    power=pd.DataFrame({'Type':'ggnull','Value':-ggnull.min(axis=1)})
    fail=pd.DataFrame({'Type':'ggnull','Value':ggnull.isnull().sum(axis=1)/d})
    
    return(power,fail)
    
def ggHelp(z,ggnullDat,k):   
    Reps=len(z)   
    p_vals=2*norm.sf(z)
    sortOrd=p_vals.argsort()
                 
    loc=ggnullDat.binEdge.searchsorted(p_vals[sortOrd])
    if max(loc)>=len(ggnullDat):
        print(k,min(p_vals),max(p_vals))
        
    ggnull=ggnullDat.ggnull.iloc[np.minimum(loc,len(ggnullDat)-1)].iloc[
        np.argsort(sortOrd)].values.astype('float32')
    
    return(k,ggnull)
        