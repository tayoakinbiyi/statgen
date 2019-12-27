import numpy as np
import pandas as pd
import pdb
from scipy.stats import norm, beta

from ail.statsPython.cpma import *
from ail.opPython.DB import *

def noCorrStats(z,pval,nameParm,parms):
    name=parms['name']
    
    N=pval.shape[1]
    pval=pval[:,0:int(N/2)]
    Reps,d=pval.shape

    Fn=(np.array([range(d)]*Reps)+1)/N
    Fn0=Fn[0]

    stats=pd.concat([pd.DataFrame({'Type':'hc','Value':[np.sqrt(N)*np.max(((Fn0 -p)/np.sqrt(p*(1-p)))[p>=1/N]) for p in pval]}),
        pd.DataFrame({'Type':'bj','Value':np.max(N*D(Fn,pval),axis=1)}),
        pd.DataFrame({'Type':'gnull','Value':np.max(-beta.cdf(pval,np.array([range(1,d+1)]*Reps),
        N-np.array([range(1,d+1)]*Reps)),axis=1)}),
        pd.DataFrame({'Type':'score','Value':np.sum(z**2,axis=1)})],axis=0)   
    
    DBLog(stats.groupby('Type').apply(lambda df: pd.DataFrame({'len':df.shape[0],'min':df['Value'].min(),'mean':df['Value'].mean(),
        'max':df['Value'].max()},index=[0])).reset_index(level=1,drop=True).to_string(),parms)

    print('noCorrStats '+nameParm,flush=True)
    
    return(stats)

def D(u,v):
    return(u*np.log(u/v)+(1-u)*np.log((1-u)/(1-v)))