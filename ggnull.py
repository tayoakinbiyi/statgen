from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil

def ggnull(z,name):
    Reps,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    
    M=multiprocessing.cpu_count()
    
    with ProcessPoolExecutor() as executor: 
        results=executor.map(ggHelp, [(i,z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)].tolist(),name)
            for i in range(int(M))])

    res=[]
    for result in results:
        res+=[result]
        
    res=sorted(res,key=lambda x: x[0])
    
    power=[]
    fail=[]
    for element in res:        
        power+=[element[1]]
        fail+=[element[2]]
        
    power=pd.concat(power,axis=0)
    fail=pd.concat(fail,axis=0)

    return(power,fail)
    
def ggHelp(dat):
    j=0;
    segment=dat[j];j+=1
    z=np.array(dat[j]);j+=1
    name=dat[j];j+=1
    
    Reps,d=z.shape
    N=int(d*2)
    
    binEdges=pd.read_csv(name+'-'+str(N)+'-ebb-binEdges.csv').bins
    ebb=pd.read_csv(name+'-'+str(N)+'-ebb.csv')

    kvec=np.array([x for x in range(d)]*Reps).flatten()
    p_vals=2*norm.sf(z).flatten()
    loc=binEdges.iloc[pd.cut(p_vals,binEdges,labels=False)+1].values+kvec

    del kvec
    del p_vals
    locOrd=loc.argsort()
    
    val=ebb.ebb.iloc[ebb.sorter.searchsorted(loc[locOrd])].reset_index(drop=True).to_frame()
    val.insert(1,'replicant',np.array([[x]*d for x in range(Reps)]).flatten()[locOrd])

    if len(val)>0:
        fail=val.groupby('replicant').apply(lambda df: pd.DataFrame({'Type':'ggnull','Value':1-df.ebb.count()/df.shape[0]},
            index=[0])).reset_index().sort_values(by='replicant')[['Type','Value']]
        power=val.groupby('replicant').apply(lambda df: pd.DataFrame({'Type':'ggnull','Value':-df.ebb.min()},index=[0])
            ).reset_index().sort_values(by='replicant')[['Type','Value']]
    else:
        fail=pd.DataFrame()
        power=pd.DataFrame()
            
    return(segment,power,fail)
    