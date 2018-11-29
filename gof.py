import numpy as np
import scipy.stats as st
import pandas as pd
import gc
from cpma import *
from ggof import *
from genL import *
from mymath import *

def monteCarlo(parms,null=False):
    alphaCSV=pd.read_csv('alpha.csv')
    alphaThere=(alphaCSV.H0==parms['H0'])&(alphaCSV.N==parms['N'])&(alphaCSV.rho1==parms['rho1'])&(alphaCSV.rho2==parms['rho2'])&(
        alphaCSV.rho3==parms['rho3'])

    if not null and sum(alphaThere)==0:
        monteCarlo(parms,True)
        
    if not null:
        alpha=alphaCSV[alphaThere]
       
    H=parms['H0'] if null else parms['H1']

    N=parms['N']
    eps=parms['eps']       
    h1_ind=range(parms['eps'])     
    arr,cr=ggStats(N)

    F_n=np.array([float(j)/N for j in range(1,N+1)])
    Stats=['ghc','hc','hcs','bj','gbj','gnull','ggnull','cpma','score','alr','fdr_bh','fdr_ratio','minP']

    L=getL(parms)
    z=np.matmul(L,st.multivariate_normal.rvs(mean=None,cov=1,size=(parms['N'],H))).T
    
    p=2*st.norm.cdf(-np.abs(z))
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    
    h_stats={}   
    for stat in Stats:
        h_stats[stat]=[]
        
    if null:
        alpha={}
        for stat in Stats:
            alpha[stat]=[]
        
    if not null:
        power={}
        for stat in Stats:
            power[stat]=[]
      
    for i in range(H):
        #if i%50==0:
        print(parms,'H',i)
        #pdb.set_trace()
            
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        cor=ggof(z[i], p_val,sig_tri,arr,cr)
        
        h_stats['minP']+=[-min(p_val)]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        h_stats['hc']+=[max(hc)] 

        hcs=(np.sqrt(N)*((F_n[:-1]-p_val[:-1])/np.sqrt(F_n[:-1]*(1-F_n[:-1]))))[cor['non_zero_hc']]
        h_stats['hcs']+=[max(hcs)] 
            
        ghc=cor['ghc']
        h_stats['ghc']+=[max(ghc)]

        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero']]
        h_stats['bj']+=[max(bj)]

        gnull=-st.beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[cor['non_zero']]
        h_stats['gnull']+=[max(gnull)]

        gbj=cor['gbj']
        h_stats['gbj']+=[max(gbj)]

        ggnull=cor['ggnull']
        h_stats['ggnull']+=[max(ggnull)]
        
        fdr_ratio=F_n/p_val
        h_stats['fdr_ratio']+=max(fdr_ratio)

        if null:
            h_stats['fdr_bh']+=[0]
        else:
            fdr_bh=F_n*.05-p_val
            h_stats['fdr_bh']+=[max(fdr_bh)]

        h_stats['cpma']+=[cpma(p_val)]

        h_stats['score']+=[sum(z[i]**2)]

        h_stats['alr']+=[sum(np.exp(np.maximum(0,D(p_val[cor['non_zero']],F_n[cor['non_zero']])))/
                np.array([2*j*np.log(N/3.0) for j in cor['non_zero']+1]))]
                
    if null:
        for stat in Stats:
            alpha[stat]=np.percentile(h_stats[stat],95)
        alpha['fdr_bh']=0

        parms.update(alpha)
        alphaCSV.append(pd.DataFrame(parms,index=[0])).to_csv('alpha.csv',index=False)
    else:
        for stat in Stats:
            power[stat]=np.mean(np.array(h1_stats[stat])>=alpha[stat])

        power=pd.DataFrame(power,index=[0]).T.reset_index()
        power.columns=['stat','value']
        power.insert(1,'type','power')
        power.index=[0]*len(power)
        power=power.merge(pd.DataFrame(parms,index=[0]),left_index=True,right_index=True)  
        
    return(power)
