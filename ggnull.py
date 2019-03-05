from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import norm, beta
import pdb
import psutil

def ggnull(z,name):
    B,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    binEdges=pd.read_csv(name+'-ebb-binEdges.csv').bins
    ebb=pd.read_csv(name+'-ebb.csv')
    
    M=multiprocessing.cpu_count()
    #results=[ggHelp((0,z,ebb,binEdges,name))]
    #pdb.set_trace()
    with ProcessPoolExecutor() as executor: 
        results=executor.map(ggHelp, [(i,z[i*int(np.ceil(B/M)):min((i+1)*int(np.ceil(B/M)),B)],ebb,binEdges,name) 
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
    z=dat[j];j+=1
    ebb=dat[j];j+=1
    binEdges=dat[j];j+=1
    name=dat[j];j+=1
    
    B,d=z.shape
    N=int(d*2)
    
    kvec=pd.Series([x for x in range(d)]*B,name='kvec')
    p_vals=pd.Series(2*norm.sf(z).flatten(),name='p_vals')
    loc=binEdges.iloc[pd.cut(p_vals,binEdges,labels=False)+1].reset_index(drop=True)+kvec

    del kvec
    del p_vals
    locOrd=loc.argsort()
    
    val=ebb.ebb.iloc[ebb.sorter.searchsorted(loc.iloc[locOrd].values)].reset_index(drop=True).to_frame()
    val.insert(1,'replicant',np.array([[x]*d for x in range(B)]).flatten()[locOrd])

    if len(val)>0:
        fail=val.groupby('replicant').apply(lambda df: pd.DataFrame({'Type':'ggnull1','Value':1-df.ebb.count()/df.shape[0]},
            index=[0])).reset_index().sort_values(by='replicant')[['Type','Value']]
        power=val.groupby('replicant').apply(lambda df: pd.DataFrame({'Type':'ggnull1','Value':-df.ebb.min()},index=[0])
            ).reset_index().sort_values(by='replicant')[['Type','Value']]
    else:
        fail=pd.DataFrame()
        power=pd.DataFrame()
            
    return(segment,power,fail)
    