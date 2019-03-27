from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import norm,beta
import psutil
import pdb

def ghc(z,name):
    Reps,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    
    M=multiprocessing.cpu_count()
    i=0
    #ghcHelp((0,z.tolist(),name))
    with ProcessPoolExecutor() as executor: 
        results=executor.map(ghcHelp, [(z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)].tolist(),name)
            for i in range(int(np.ceil(Reps/np.ceil(Reps/M))))])

    res=[]
    for result in results:
        res+=[result]
    
    power=pd.concat(res,axis=0)
    return(power)

def ghcHelp(dat):
    j=0
    z=np.array(dat[j]);j+=1
    name=dat[j];j+=1
    
    Reps,d=z.shape
    N=int(d*2)
    
    var=pd.read_csv('ebb/'+name+'/var.csv')

    kvec=np.array([range(d)]*Reps).flatten()
    p_vals=2*norm.sf(z).flatten()
    sortOrd=np.argsort(p_vals)

    val=pd.DataFrame({'var':var['var'].iloc[var.binEdges.searchsorted(p_vals[sortOrd]).flatten()],
        'k':kvec[sortOrd],'replicant':np.array([range(Reps)]*d).T.flatten()[sortOrd],'p':p_vals[sortOrd]})
    
    if(len(val.replicant.drop_duplicates())<len(z)):
        print(len(val.replicant.drop_duplicates()),len(z))
    
    power=val.groupby('replicant').apply(lambda df,N: pd.DataFrame({'Type':'ghcFull','Value':
        np.max(((df.k+1-N*df.p)/(df['var']**.5)))},index=[0]),N=N).reset_index().sort_values(by='replicant')[['Type','Value']]

    val=val[val.p>=1/N]
    
    power.append(val.groupby('replicant').apply(lambda df,N: pd.DataFrame({'Type':'ghc','Value':
        np.max(((df.k+1-N*df.p)/(df['var']**.5)))},index=[0]),N=N).reset_index().sort_values(
        by='replicant')[['Type','Value']])
    
    return(power)

