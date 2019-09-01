from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import norm,beta
import psutil
import pdb

def ghc(z,var,parms):
    cpus=parms['cpus']
    Reps,N=z.shape
    delta=parms['delta']
    
    d=int(np.ceil(N*delta))

    z=-np.sort(-np.abs(z))[:,0:d]
    
    futures=[]
    with ProcessPoolExecutor(cpus) as executor: 
        for i in range(int(np.ceil(Reps/np.ceil(Reps/cpus)))):
            futures.append(executor.submit(ghcHelp,z[i*int(np.ceil(Reps/cpus)):min((i+1)*int(np.ceil(Reps/cpus)),Reps)],var,parms))

    ghc=pd.DataFrame(dtype='float32')
    fail=pd.DataFrame(dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        ghc=ghc.append(f.result()[0])
        fail=fail.append(f.result()[1])
    
    return(ghc,fail)

def ghcHelp(z,var,parms):
    Reps,d=z.shape
    N=parms['N']
    
    kvec=np.array([range(d)]*Reps).flatten()
    p_vals=2*norm.sf(z).flatten()
    sortOrd=np.argsort(p_vals)

    loc=var.binEdge.searchsorted(p_vals[sortOrd]).flatten()
        
    val=pd.DataFrame({'var':var['var'].iloc[loc],
        'k':kvec[sortOrd],'replicant':np.array([range(Reps)]*d).T.flatten()[sortOrd],'p':p_vals[sortOrd]})
    
    if(len(val.replicant.drop_duplicates())<len(z)):
        print(len(val.replicant.drop_duplicates()),len(z))
    
    power=val.groupby('replicant').apply(lambda df,N: pd.DataFrame({'Type':'ghcFull','Value':
        np.max(((df.k+1-N*df.p)/(df['var']**.5)))},index=[0]),N=N).reset_index().sort_values(by='replicant')[['Type','Value']]

    val=val[val.p>=1/N]
    
    fail=val.groupby('replicant').apply(lambda df,d: pd.DataFrame({'Type':'ghc', 'Value':sum(pd.isnull(df['var']))/d},index=[0]),
        d=d).reset_index().sort_values(by='replicant')[['Type','Value']]
    power=power.append(val.groupby('replicant').apply(lambda df,N: pd.DataFrame({'Type':'ghc','Value':
        np.nanmax(((df.k+1-N*df.p)/(df['var']**.5)))},index=[0]),N=N).reset_index().sort_values(by='replicant')[['Type','Value']])
    
    return(power,fail)

