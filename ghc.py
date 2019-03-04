from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import norm,beta

def ghc(z,name):
    B,N=z.shape
    d=int(N/2)

    z=-np.sort(-np.abs(z))[:,0:d]
    binEdges=pd.read_csv(name+'-ebb-binEdges.csv').bins
    ebb=pd.read_csv(name+'-ebb.csv')
    
    M=multiprocessing.cpu_count()
    results=[ghcHelp((0,z,ebb,binEdges,name))]
    #with ProcessPoolExecutor() as executor: 
    #    results=executor.map(ghcHelp, [(i,z[i*int(np.ceil(len(z)/M)):min((i+1)*int(np.ceil(len(z)/M)),len(z))],ebb,binEdges,name) 
    #        for i in range(int(M))])

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
    z=dat[j];j+=1
    ebb=dat[j];j+=1
    binEdges=dat[j];j+=1
    name=dat[j];j+=1
    
    B,d=z.shape
    N=int(d*2)
    
    kvec=pd.Series([x for x in range(d)]*B,name='kvec')
    replicant=pd.Series([x for x in range(B) for i in range(d)]*B,name='replicant')
    p_vals=pd.Series(2*norm.sf(z).flatten(),name='p_vals')
    
    loc=binEdges.iloc[pd.cut(p_vals,binEdges,labels=False)+1].reset_index(drop=True)+kvec
    locOrd=loc.argsort()
    
    val=ebb[['var','k','sorter']].iloc[ebb.sorter.searchsorted(loc.iloc[locOrd].values)].reset_index(drop=True)
    val.insert(2,'replicant',replicant.iloc[locOrd].values)
    val.insert(3,'p',p_vals.iloc[locOrd].values)
    val=val[val.p>=1/N]
    
    if len(val)>0:
        power=val.groupby('replicant').apply(lambda df,N: pd.DataFrame({'Type':'ghc1','Value':np.max((df.k+1-N*df.p)/(df['var']**.5))},
            index=[0]),N=N).reset_index().sort_values(by='replicant')[['Type','Value']]
        ghc1=val.reset_index().sort_values(by=['replicant','k'])
        ghc1.insert(0,'Type','ghc1')
        ghc1.insert(0,'Value',(ghc1.k+1-N*ghc1.p)/(ghc1['var']**.5))
        #pdb.set_trace()
        ghc1=ghc1.loc[(ghc1.p>=1/N),['Type','Value','replicant','k','p','var','sorter']]
    else:
        power=pd.DataFrame()
    
    return(segment,power,ghc1)

