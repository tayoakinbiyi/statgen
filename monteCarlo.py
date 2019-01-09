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


def monteCarlo(parms,stats,L,nullHyp=False):
    np.random.seed(1)
    freeze_support()

    alphaCSV=pd.read_csv('alpha.csv')
    alphaThere=(alphaCSV.H0==parms['H0'])&(alphaCSV.N==parms['N'])&(alphaCSV.sig==parms['sig'])

    if not nullHyp and sum(alphaThere)==0:
        alpha=monteCarlo(parms,stats,L,True).reset_index()
    elif not nullHyp:
        alpha=alphaCSV[alphaThere].reset_index()
       
    H=parms['H0'] if nullHyp else parms['H1']

    N=parms['N']
    arr,cr=ggStats(N)
    print(parms,nullHyp)

    F_n=np.array([float(j)/N for j in range(1,N+1)])

    z=np.matmul(L,st.multivariate_normal.rvs(mean=0,cov=1,size=(parms['N'],H))).T
    if not nullHyp:
        z[:,random.sample(range(z.shape[1]),parms['eps'])]+=parms['mu']
    
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    
    h_stats={}   
    for stat in stats:
        h_stats[stat]=[]
        
    M=float(cpu_count())
    #i=0
    #mc((parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,sig_tri,nullHyp,L))
    #pdb.set_trace()
        
    try:
        pool = Pool(cpu_count())
        results=pool.map(mc, [(parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,sig_tri,nullHyp) for i in range(int(M))])
    finally:
        pool.close()
        pool.join()            

    res=pd.DataFrame()
    for result in results:
        res=res.append(result)
                        
    if 'alpha' not in locals():
        alpha=pd.DataFrame(res).apply(np.nanpercentile,q=95).to_frame().T.merge(pd.DataFrame(
            parms,index=[0]),left_index=True,right_index=True)
        alphaCSV.append(alpha).to_csv('alpha.csv',index=False)
        print(sig_tri[0:10].round(2),np.mean(sig_tri))
        return(alpha)    
    else:
        res.index=[0]*len(res)
        power=(res-alpha[stats]>=0).apply(np.nanmean).to_frame().T.merge(pd.DataFrame(
            parms,index=[0]),left_index=True,right_index=True)  
        return(power)

def mc(data):
    j=0
    parms=data[j];j+=1
    F_n=data[j];j+=1
    arr=data[j];j+=1
    cr=data[j];j+=1
    z=data[j];j+=1
    h_stats=data[j];j+=1
    sig_tri=data[j];j+=1
    nullHyp=data[j];j+=1
    
    p=2*st.norm.cdf(-np.abs(z))  
    stats=h_stats.keys()
    N=parms['N']

    for i in range(len(p)):     
        #print(i)
        #pdb.set_trace()
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        
        cor=ggof(z[i], p_val,sig_tri,arr,cr)
        
        h_stats['minP']+=[np.max(-np.log(p_val))]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        h_stats['hc']+=[np.max(hc) if len(hc)>0 else np.nan] 

        h_stats['ghc']+=[np.max(cor['ghc']) if len(cor['ghc']>0) else np.nan]
        
        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero']]
        h_stats['bj']+=[np.max(bj) if len(bj)>0 else np.nan]
            
        h_stats['gbj']+=[np.max(cor['gbj']) if len(cor['gbj'])>0 else np.nan]

        gnull=-st.beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[cor['non_zero']]
        h_stats['gnull']+=[np.max(gnull)]
        
        h_stats['ggnull']+=[np.max(cor['ggnull']) if len(cor['ggnull']>0) else np.nan]
        
        fdr_ratio=F_n/p_val
        h_stats['fdr_ratio']+=[np.max(fdr_ratio)]

        h_stats['cpma']+=[cpma(p_val)]

        h_stats['score']+=[np.sum(z[i]**2)]

        h_stats['alr']+=[np.sum(np.exp(np.maximum(0,D(p_val[cor['non_zero']],F_n[cor['non_zero']])))/
                np.array([2*j*np.log(N/3.0) for j in cor['non_zero']+1]))]
  
    return(pd.DataFrame(h_stats))