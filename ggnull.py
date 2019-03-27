from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil

def ggnull(z,sigName):
    Reps,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    
    M=multiprocessing.cpu_count()
    
    ebb=np.loadtxt('ebb/'+sigName+'/ebb.csv',delimiter=',')
    minMaxB=np.loadtxt('ebb/'+sigName+'/minMaxB.csv',delimiter=',').astype(int)
        
    k=0
    #ggHelp((z[:,k].tolist(),ebb[minMaxB[k,0]:minMaxB[k,1]].tolist()))
    #pdb.set_trace()
    with ProcessPoolExecutor() as executor: 
        results=executor.map(ggHelp, [(z[:,k].flatten().tolist(),ebb[minMaxB[k,0]:minMaxB[k,1]].tolist()) for k in range(d)])
    
    res=[]
    for result in results:
        res+=[result]
               
    raw=np.concatenate(res)
    
    power=pd.DataFrame({'Type':'ggnull','Value':-np.min(raw,axis=0)})
    fail=pd.DataFrame({'Type':'ggnull','Value':np.sum(np.isnan(raw),axis=0)})

    return(power,fail)
    
def ggHelp(dat):
    j=0;
    z=np.array(dat[j]);j+=1
    ebb=pd.DataFrame(dat[j],columns=['binEdge','ebb']);j+=1
    
    Reps=len(z)   
    p_vals=2*norm.sf(z)
    sortOrd=p_vals.argsort()
                          
    raw=ebb.ebb.iloc[ebb.binEdge.searchsorted(p_vals[sortOrd])].values.flatten()[np.argsort(sortOrd)].reshape(1,-1).tolist()
    
    return(raw)
    