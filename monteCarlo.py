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


def monteCarlo(parms,stats,L,nullHyp=False):
    np.random.seed(1)
    freeze_support()

    alphaCSV=pd.read_csv('alpha.csv')
    whichParms=['H0','N','min_cor','avg_cor','max_cor']
    
    alphaThere=[x==pd.Series(parms)[whichParms].values.tolist() for x in alphaCSV[whichParms].values.tolist()]
    
    if not nullHyp and sum(alphaThere)==0:
        alpha=monteCarlo(parms,stats,L,True).reset_index()
    elif not nullHyp:
        alpha=alphaCSV[alphaThere].reset_index()
       
    H=parms['H0'] if nullHyp else parms['H1']

    N=parms['N']
    arr,cr=ggStats(N)
    print(parms,nullHyp)

    F_n=np.array([float(j)/N for j in range(1,N+1)])

    z=np.matmul(L.T,st.multivariate_normal.rvs(mean=0,cov=1,size=(parms['N'],H))).T
    #if not nullHyp:
    #    z[:,random.sample(range(z.shape[1]),parms['eps'])]+=parms['mu']
    
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    
    h_stats={}   
    fail={'ghc':[],'gbj':[],'ggnull':[]}
    for stat in stats:
        h_stats[stat]=[]
        
    M=float(cpu_count())
    #i=0
    #mc((parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,fail,sig_tri,nullHyp))
    #pdb.set_trace()
        
    try:
        pool = Pool(cpu_count())
        results=pool.map(mc, [(parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,fail,sig_tri,nullHyp) for i in range(int(M))])
    finally:
        pool.close()
        pool.join()            

    power=pd.DataFrame()
    fail=pd.DataFrame()
    for result in results:
        power=power.append(result[0])
        fail=fail.append(result[1])
                     
    t_parms=parms.copy()
    for col in fail.columns:
        t_parms[col+'-AvgFailPct']=fail[col].mean()
        t_parms[col+'-PctAllFail']=(fail[col]==1).mean()

    if 'alpha' not in locals():
        alpha=pd.DataFrame(power).apply(np.nanpercentile,q=95).to_frame().T.merge(pd.DataFrame(
            t_parms,index=[0]),left_index=True,right_index=True)
        alpha.drop(columns=['mu','eps'],inplace=True)
        alphaCSV.append(alpha,sort=False).to_csv('alpha.csv',index=False)
        return(alpha)    
    else:
        power.index=[0]*len(power)
        power=(power-alpha[stats]>=0).apply(np.nanmean).to_frame().T.merge(pd.DataFrame(
            t_parms,index=[0]),left_index=True,right_index=True)  
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
    stats=h_stats.keys()
    N=parms['N']

    for i in range(len(p)):     
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        
        cor=ggof(z[i], p_val,sig_tri,arr,cr)

        h_stats['minP']+=[np.max(-np.log(p_val))]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        h_stats['hc']+=[np.max(hc)] 

        fail['ghc']+=[1-float(len(cor['ghc']))/len(cor['non_zero_hc'])]
        h_stats['ghc']+=[np.max(cor['ghc']) if len(cor['ghc']>0) else np.nan if nullHyp else 0]
        
        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero']]
        h_stats['bj']+=[np.max(bj)]
            
        fail['gbj']+=[1-float(len(cor['gbj']))/len(cor['non_zero'])]
        h_stats['gbj']+=[np.max(cor['gbj']) if len(cor['gbj'])>0 else np.nan if nullHyp else 0]

        gnull=-st.beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[cor['non_zero']]
        h_stats['gnull']+=[np.max(gnull)]
        
        fail['ggnull']+=[1-float(len(cor['ggnull']))/len(cor['non_zero'])]
        h_stats['ggnull']+=[np.max(cor['ggnull']) if len(cor['ggnull']>0) else np.nan if nullHyp else -1]
        
        fdr_ratio=F_n/p_val
        h_stats['fdr_ratio']+=[np.max(fdr_ratio)]

        h_stats['score']+=[np.sum(z[i]**2)]

    return(pd.DataFrame(h_stats),pd.DataFrame(fail))