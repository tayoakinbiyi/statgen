import numpy as np
import scipy.stats as st
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support
import random
import pdb

from ggof import *
from mymath import *


def monteCarlo(H,N,mu,eps,sigName,nullHyp,L):
    np.random.seed(1)
    freeze_support()

    arr,cr=ggStats(N)
    print(N,mu,eps,sigName,nullHyp)

    F_n=np.array([float(j)/N for j in range(1,N+1)])

    z=np.matmul(L.T,st.multivariate_normal.rvs(mean=0,cov=1,size=(N,H))).T
    if not nullHyp:
        z[:,random.sample(range(z.shape[1]),eps)]+=mu
    
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    
    M=float(cpu_count())
    #i=0
    #results=mc((N,nullHyp,F_n,arr,cr,z[i*int(np.ceil(H/M)):min((i+1)*int(np.ceil(H/M)),H)],sig_tri))
    #pdb.set_trace()
        
    try:
        pool = Pool(cpu_count())
        results=pool.map(mc, [(N,nullHyp,F_n,arr,cr,z[i*int(np.ceil(H/M)):min((i+1)*int(np.ceil(H/M)),H)],sig_tri) for i in range(int(M))])
    finally:
        pool.close()
        pool.join()            

    power=pd.DataFrame()
    for result in results:
        power=power.append(result)
                     
    power.columns=['Type','Value']
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[mu,eps,nullHyp]],columns=['mu','eps','nullHyp'],index=[0]),left_index=True,right_index=True)

    return(power)

def mc(data):
    j=0
    N=data[j];j+=1
    nullHyp=data[j];j+=1
    F_n=data[j];j+=1
    arr=data[j];j+=1
    cr=data[j];j+=1
    z=data[j];j+=1
    sig_tri=data[j];j+=1
    
    p=2*st.norm.cdf(-np.abs(z))  
    
    out=[]

    for i in range(len(p)):     
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        
        cor=ggof(z[i], p_val,sig_tri,arr,cr)

        out+=[['minP',np.max(-np.log(p_val[cor['non_zero']]))]]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        out+=[['hc',np.max(hc)]]

        out+=[['ghc-fail',1-float(len(cor['ghc']))/len(cor['non_zero_hc'])]]
        out+=[['ghc',np.max(cor['ghc']) if len(cor['ghc']>0) else np.nan if nullHyp else 0]]
        
        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero']]
        out+=[['bj',np.max(bj)]]
            
        out+=[['gbj-fail',1-float(len(cor['gbj']))/len(cor['non_zero'])]]
        out+=[['gbj',np.max(cor['gbj']) if len(cor['gbj'])>0 else np.nan if nullHyp else 0]]

        gnull=-st.beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[cor['non_zero']]
        out+=[['gnull',np.max(gnull)]]
        
        out+=[['ggnull-fail',1-float(len(cor['ggnull']))/len(cor['non_zero'])]]
        out+=[['ggnull',np.max(cor['ggnull']) if len(cor['ggnull']>0) else np.nan if nullHyp else -1]]
        
        fdr_ratio=(F_n/p_val)[cor['non_zero']]
        out+=[['fdr_ratio',np.max(fdr_ratio)]]

        out+=[['score',np.sum(z[i]**2)]]

    return(pd.DataFrame(out))