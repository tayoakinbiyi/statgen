from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil

def ggnull(z,sigName,ebb):
    Reps,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    
    M=multiprocessing.cpu_count()
    
    minMaxB=pd.read_csv('ebb/'+sigName+'/minMaxB.csv').astype(int)

    futures=[]
    with ProcessPoolExecutor() as executor: 
        for k in range(d):
            futures.append(executor.submit(ggHelp,z[:,k],ebb.iloc[minMaxB.loc[k,'start']:minMaxB.loc[k,'end']],k))
    
    ggnull=pd.DataFrame(dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        result=f.result()
        ggnull.insert(ggnull.shape[1],result[0],result[1])
    
    power=pd.DataFrame({'Type':'ggnull','Value':-ggnull.min(axis=1)})
    fail=pd.DataFrame({'Type':'ggnull','Value':ggnull.isnull().sum(axis=1)/d})
    
    return(power,fail)
    
def ggHelp(z,ebb,k):   
    Reps=len(z)   
    p_vals=2*norm.sf(z)
    sortOrd=p_vals.argsort()
                 
    ggnull=ebb.ebb.iloc[np.minimum(ebb.binEdge.searchsorted(p_vals[sortOrd]),len(ebb)-1)].iloc[
        np.argsort(sortOrd)].values.astype('float32')
    
    return(k,ggnull)
        