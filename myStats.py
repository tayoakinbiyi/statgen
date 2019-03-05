import pandas as pd
import numpy as np
from scipy.stats import norm, beta

from myMath import *

def myStats(z_raw):
    B,N=z_raw.shape
    d=int(N/2)
    
    z=-np.sort(-np.abs(z_raw))[:,0:d]
    p_val=2*norm.sf(z)

    Fn=(np.array([range(d)]*B)+1)/N
    
    minP=-np.min(np.log(p_val),axis=1)    
    hc=np.array([np.sqrt(N)*np.max([((k+1)/N -p[k])/np.sqrt(p[k]*(1-p[k])) for k in range(d) if p[k]>=1/N]) for p in p_val.tolist()])
    bj=np.max(N*D(Fn,p_val),axis=1)
    gnull=np.max(-beta.cdf(p_val,np.array([range(1,d+1)]*B),(N+1)-np.array([range(1,d+1)]*B)),axis=1)
    fdr=np.max(Fn/p_val,axis=1)
    score=np.sum(z**2,axis=1)
    
    power=pd.concat([
        pd.DataFrame({'Type':'hc1','Value':hc}),
        pd.DataFrame({'Type':'minP1','Value':minP}),
        pd.DataFrame({'Type':'bj1','Value':bj}),
        pd.DataFrame({'Type':'gnull1','Value':gnull}),
        pd.DataFrame({'Type':'fdr1','Value':fdr}),
        pd.DataFrame({'Type':'score1','Value':score})
    ],axis=0)
    return(power)

