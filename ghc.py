from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import norm,beta
import psutil

def ghc(z,name):
    Reps,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    
    M=multiprocessing.cpu_count()
    #results=[ghcHelp((0,z,ebb,binEdges,name))]
    with ProcessPoolExecutor() as executor: 
        results=executor.map(ghcHelp, [(i,z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)].tolist(),name)
        for i in range(int(M))])

    res=[]
    for result in results:
        res+=[result]
    
    res=sorted(res,key=lambda x: x[0])
    
    power=[]
    for element in res:        
        power+=[element[1]]
        
    power=pd.concat(power,axis=0)
    return(power)

def ghcHelp(dat):
    j=0
    segment=dat[j];j+=1
    z=np.array(dat[j]);j+=1
    name=dat[j];j+=1
    
    Reps,d=z.shape
    N=int(d*2)
    
    binEdges=pd.read_csv(name+'-'+str(N)+'-ebb-binEdges.csv').bins
    ebb=pd.read_csv(name+'-'+str(N)+'-ebb.csv')

    kvec=np.array([range(d)]*Reps).flatten()
    p_vals=2*norm.sf(z).flatten()
    loc=binEdges.iloc[pd.cut(p_vals,binEdges,labels=False)+1].values+kvec

    locOrd=loc.argsort()
    
    val=ebb[['var','k','sorter']].iloc[ebb.sorter.searchsorted(loc[locOrd])].reset_index(drop=True)
    val.insert(2,'replicant',np.array([[x]*d for x in range(Reps)]).flatten()[locOrd])
    val.insert(3,'p',p_vals[locOrd])
    
    del p_vals
    val=val[val.p>=1/N]
    
    if len(val)>0:
        power=val.groupby('replicant').apply(lambda df,N: pd.DataFrame({'Type':'ghc','Value':np.max((df.k+1-N*df.p)/(df['var']**.5))},
            index=[0]),N=N).reset_index().sort_values(by='replicant')[['Type','Value']]
    else:
        power=pd.DataFrame()
    
    return(segment,power)

