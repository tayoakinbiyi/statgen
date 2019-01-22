import numpy as np
import scipy.stats as st
import pandas as pd
import gc
from cpma import *
from ggof import *
from genL import *
from mymath import *
from multiprocessing import Pool, cpu_count, freeze_support
import random
import pdb


def monteCarlo(t_parms,stats,L,nullHyp=False):
    stats=stats+['fdr_arg']
    np.random.seed(1)
    freeze_support()

    H=parms['H0'] if nullHyp else parms['H1']

    N=parms['N']
    arr,cr=ggStats(N)
    print(parms,nullHyp)

    F_n=np.array([float(j)/N for j in range(1,N+1)])

    z=np.matmul(L.T,st.multivariate_normal.rvs(mean=0,cov=1,size=(parms['N'],H))).T
    if not nullHyp:
        z[:,random.sample(range(z.shape[1]),parms['eps'])]+=parms['mu']
    
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    
    h_stats={}   
    fail={'ghc':[],'gbj':[],'ggnull':[]}
    for stat in stats:
        h_stats[stat]=[]
        
    M=float(cpu_count())
    i=0
    mc((parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,fail,sig_tri,nullHyp))
    pdb.set_trace()
        
    try:
        pool = Pool(cpu_count())
        results=pool.map(mc, [(parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,fail,sig_tri,nullHyp) for i in range(int(M))])
    finally:
        pool.close()
        pool.join()            

    power=pd.DataFrame()
    for result in results:
        power=power.append(result[0])
                     
    power.index=pd.MultiIndex.from_tuples(len(power)*[(parms['sigName'],parms['mu'] if not nullHyp else np.nan,parms['eps']
        if not nullHyp else np.nan,nullHyp)],names=['sigName','mu','eps','nullHyp'])

    return(power)

def mc(data):
    j=0
    parms=data[j];j+=1
    F_n=data[j];j+=1
    arr=data[j];j+=1
    cr=data[j];j+=1
    z=data[j];j+=1
    h_stats=data[j];j+=1
    fail=data[j];j+=1
    sig_tri=data[j];j+=1
    nullHyp=data[j];j+=1
    
    p=2*st.norm.cdf(-np.abs(z))  
    N=parms['N']

    for i in range(len(p)):     
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        
        cor=ggof(z[i], p_val,sig_tri,arr,cr)

        h_stats['minP']+=[np.max(-np.log(p_val[cor['non_zero']]))]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        h_stats['hc']+=[np.max(hc)] 

        h_stats['ghc-fail']+=[1-float(len(cor['ghc']))/len(cor['non_zero_hc'])]
        h_stats['ghc']+=[np.max(cor['ghc']) if len(cor['ghc']>0) else np.nan if nullHyp else 0]
        
        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero']]
        h_stats['bj']+=[np.max(bj)]
            
        h_stats['gbj-fail']+=[1-float(len(cor['gbj']))/len(cor['non_zero'])]
        h_stats['gbj']+=[np.max(cor['gbj']) if len(cor['gbj'])>0 else np.nan if nullHyp else 0]

        gnull=-st.beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[cor['non_zero']]
        h_stats['gnull']+=[np.max(gnull)]
        
        h_stats['ggnull-fail']+=[1-float(len(cor['ggnull']))/len(cor['non_zero'])]
        h_stats['ggnull']+=[np.max(cor['ggnull']) if len(cor['ggnull']>0) else np.nan if nullHyp else -1]
        
        fdr_ratio=(F_n/p_val)[cor['non_zero']]
        h_stats['fdr_ratio']+=[np.max(fdr_ratio)]

        h_stats['score']+=[np.sum(z[i]**2)]

    return(pd.DataFrame(h_stats),pd.DataFrame(fail))