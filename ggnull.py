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
    
    i=0
    #ggHelp((i,z.tolist(),name))
    with ProcessPoolExecutor() as executor: 
        results=executor.map(ggHelp, [(i,z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)].tolist(),name)
            for i in range(int(Reps/np.ceil(Reps/M)))])
    
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
    ebb=pd.read_csv(name+'-'+str(N)+'-ebb-prob.csv')

    kvec=np.array([range(d)]*Reps).flatten()
    p_vals=2*norm.sf(z).flatten()
    sorter=binEdges.iloc[np.minimum(len(binEdges)-1,np.digitize(p_vals,binEdges))].values+kvec

    sortOrd=sorter.argsort()
    
    val=pd.DataFrame()
    val.insert(0,'ebb',ebb.ebb.iloc[ebb.sorter.searchsorted(sorter[sortOrd])].values)
    val.insert(1,'replicant',np.array([range(Reps)]*d).T.flatten()[sortOrd])

    if len(val)>0:
        fail=val.groupby('replicant').apply(lambda df: pd.DataFrame({'Type':'ggnull','Value':1-df.ebb.count()/df.shape[0]},
            index=[0])).reset_index().sort_values(by='replicant')[['Type','Value']]
        power=val.groupby('replicant').apply(lambda df: pd.DataFrame({'Type':'ggnull','Value':-df.ebb.min()},index=[0])
            ).reset_index().sort_values(by='replicant')[['Type','Value']]
    else:
        fail=pd.DataFrame()
        power=pd.DataFrame()
    
    return(segment,power,fail)
    