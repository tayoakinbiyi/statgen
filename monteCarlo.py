import numpy as np
import scipy.stats as st
import pandas as pd
import gc
from cpma import *
from ggof import *
from genL import *
from mymath import *
from multiprocessing import Pool, cpu_count, freeze_support


def monteCarlo(parms,null=False):
    np.random.seed(1)
    freeze_support()

    alphaCSV=pd.read_csv('alpha.csv')
    alphaThere=(alphaCSV.H0==parms['H0'])&(alphaCSV.N==parms['N'])&(alphaCSV.rho1==parms['rho1'])&(alphaCSV.rho2==parms['rho2'])&(
        alphaCSV.rho3==parms['rho3'])

    if not null and sum(alphaThere)==0:
        alpha=monteCarlo(parms,True).reset_index()
    elif not null:
        alpha=alphaCSV[alphaThere].reset_index()
       
    H=parms['H0'] if null else parms['H1']

    N=parms['N']
    arr,cr=ggStats(N)
    print(parms,null)

    F_n=np.array([float(j)/N for j in range(1,N+1)])
    stats=['ghc','hc','hcs','bj','gbj','gnull','ggnull','cpma','score','alr','fdr_bh','fdr_ratio','minP']

    L=getL(parms)
    z=np.matmul(L,st.multivariate_normal.rvs(mean=None,cov=1,size=(parms['N'],H))).T
    if not null:
        z[:,:parms['eps']]+=parms['mu']
    
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    
    h_stats={}   
    for stat in stats:
        h_stats[stat]=[]
        
    if null:
        alpha={}
        for stat in stats:
            alpha[stat]=[]
    else:
        power={}
        for stat in stats:
            power[stat]=[]
      
    M=1#cpu_count()
    for i in range(M):
        mc((parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,sig_tri,null))
        
    try:
        pool = Pool(1)#cpu_count())
        results=pool.map(mc, [(parms,F_n,arr,cr,z[i*int(H/M):min((i+1)*int(H/M),H)],h_stats,sig_tri,null) for i in range(M)])
    finally:
        pool.close()
        pool.join()            

    res=pd.DataFrame()
    for result in results:
        res=res.append(result)
                        
    if null:
        alpha=pd.DataFrame(res).apply(np.percentile,q=95).to_frame().T.merge(pd.DataFrame(
            parms,index=[0]),left_index=True,right_index=True)
        alphaCSV.append(alpha).to_csv('alpha.csv',index=False)
        return(alpha)
    
    else:
        res.index=[0]*len(res)
        power=(res-alpha[stats]>=0).apply(np.mean).to_frame().T.merge(pd.DataFrame(
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
    null=data[j];j+=1
    
    p=2*st.norm.cdf(-np.abs(z))  
    stats=h_stats.keys()
    N=parms['N']

    for i in range(len(p)):     
        print(i)
        #pdb.set_trace()
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        
        cor=ggof(z[i], p_val,sig_tri,arr,cr)
        
        h_stats['minP']+=[max(-np.log(p_val))]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        h_stats['hc']+=[max(hc)] 

        hcs=(np.sqrt(N)*((F_n[:-1]-p_val[:-1])/np.sqrt(F_n[:-1]*(1-F_n[:-1]))))[cor['non_zero_hc']]
        h_stats['hcs']+=[max(hcs)] 
            
        ghc=cor['ghc']
        h_stats['ghc']+=[max(ghc)]

        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero_gbj']]
        h_stats['bj']+=[max(bj)]

        gnull=-st.beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[cor['non_zero']]
        h_stats['gnull']+=[max(gnull)]
        
        gbj=cor['gbj']
        h_stats['gbj']+=[max(gbj)]

        ggnull=cor['ggnull']
        h_stats['ggnull']+=[max(ggnull)]
        
        fdr_ratio=F_n/p_val
        h_stats['fdr_ratio']+=[max(fdr_ratio)]

        if null:
            h_stats['fdr_bh']+=[0]
        else:
            fdr_bh=F_n*.05-p_val
            h_stats['fdr_bh']+=[max(fdr_bh)]

        h_stats['cpma']+=[cpma(p_val)]

        h_stats['score']+=[sum(z[i]**2)]

        h_stats['alr']+=[sum(np.exp(np.maximum(0,D(p_val[cor['non_zero']],F_n[cor['non_zero']])))/
                np.array([2*j*np.log(N/3.0) for j in cor['non_zero']+1]))]
    
    return(pd.DataFrame(h_stats))