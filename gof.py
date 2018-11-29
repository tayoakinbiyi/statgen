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
    pdb.set_trace()
    alphaThere=(alphaCSV.H0==parms['H0'])&(alphaCSV.N==parms['N'])&(alphaCSV.rho1==parms['rho1'])&(alphaCSV.rho2==parms['rho2'])&(
        alphaCSV.rho3==parms['rho3'])

    if not null and sum(alphaThere)==0:
        monteCarlo(parms,True)
        
    if not null:
        alpha=alphaCSV[alphaThere]
       
    H=parms['H0'] if null else parms['H1']

    z=st.multivariate_normal.rvs(mean=None,cov=1,size=(parms['N'],H))
    p=2*st.norm.cdf(-np.abs(z))

    N=parms['N']
    eps=parms['eps']       
    h1_ind=range(parms['eps'])     
    arr,cr=ggStats(N)

    F_n=np.array([float(j)/N for j in range(1,N+1)])
    Stats=['ghc','hc','hcs','bj','gbj','gnull','ggnull','cpma','score','alr','fdr_bh','fdr_ratio','minP']
    L=getL(parms)
    sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
    pdb.set_trace()
    
    h_stats={}   
    for stat in Stats:
        h_stats[stat]=[]
        
    if null:
        alpha={}
        for stat in Stats:
            alpha[stat]=[]
        
    if not null:
        power={}
        recall={}
        precision={}
        for stat in Stats:
            power[stat]=[]
            recall[stat]=[]
            precision[stat]=[]
      
    for i in range(H):
        if i%50==0:
            print(parms,'H',i)
            
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        cor=ggof(z[i], p_val,sig_tri,arr,cr)

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[cor['non_zero_hc']]
        h_stats['hc']+=[max(hc)] 

        hcs=(np.sqrt(N)*((F_n[:-1]-p_val[:-1])/np.sqrt(F_n[:-1]*(1-F_n[:-1]))))[cor['non_zero_hc']]
        h_stats['hcs']+=[max(hcs)] 
            
        ghc=cor['ghc']
        h_stats['ghc']+=[max(ghc)]

        bj=N*(D(F_n[:-1],p_val[:-1]))[cor['non_zero']]
        h_stats['bj']+=[bj[bj_max-1]]

        rbj=N*(D(p_val[:-1],F_n[:-1]))[cor['non_zero']]
        h_stats['rbj']+=[rbj[rbj_max-1]]

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
            fdr_bh=np.array([float(j*fdr_bh_q)/N for j in range(1,N+1)])-p_val
            h_stats['fdr_bh']+=[max(fdr_bh)]

        h_stats['cpma']+=[cpma(p_val)]

        h_stats['score']+=[sum(z[i]**2)]

        h_stats['alr']+=[sum(np.exp(np.maximum(0,D(p_val[cor['non_zero']],F_n[cor['non_zero']])))/
                np.array([2*j*np.log(N/3.0) for j in cor['non_zero']+1]))]
            
        if not null:
            for stat in ['hcs']:
                if h_stats[stat]>alpha[stat]:
                    hcs_max=(cor['non_zero_hc']+1)[np.argmax(hcs)]
            precision['hcs']+=[float(len(set(p_val_ind[0:hcs_max])&set(h_ind)))/hcs_max]
            recall['hcs']+=[float(len(set(p_val_ind[0:hcs_max])&set(h_ind)))/eps]

            hc_max=(cor['non_zero_hc']+1)[np.argmax(hc)]
            precision['hc']+=[float(len(set(p_val_ind[0:hc_max])&set(h_ind)))/hc_max]
            recall['hc']+=[float(len(set(p_val_ind[0:hc_max])&set(h_ind)))/eps]
            
            if len(cor['non_zero_ghc'])>0:
                ghc_max=(cor['non_zero_ghc']+1)[np.argmax(ghc)]
                precision['ghc']+=[float(len(set(p_val_ind[0:ghc_max])&set(h_ind)))/ghc_max]
                recall['ghc']+=[float(len(set(p_val_ind[0:ghc_max])&set(h_ind)))/eps] 
            else:
                precision['ghc']+=[0]
                recall['ghc']+=[0]

            bj_max=np.argmax(bj)+1
            precision['bj']+=[float(len(set(p_val_ind[0:bj_max])&set(h_ind)))/bj_max]
            recall['bj']+=[float(len(set(p_val_ind[0:bj_max])&set(h_ind)))/eps]

            rbj_max=np.argmax(rbj)+1
            precision['rbj']+=[float(len(set(p_val_ind[0:rbj_max])&set(h_ind)))/rbj_max]
            recall['rbj']+=[float(len(set(p_val_ind[0:rbj_max])&set(h_ind)))/eps]

            if h_stats['gnull'][i]>=alpha['gnull']:
                gnull_max=(cor['non_zero']+1)[min(np.where(gnull>=alpha['gnull'])[0])]
                precision['gnull']+=[float(len(set(p_val_ind[0:gnull_max])&set(h_ind)))/gnull_max]            
                recall['gnull']+=[float(len(set(p_val_ind[0:gnull_max])&set(h_ind)))/eps]         
            else:
                precision['gnull']+=[0]
                recall['gnull']+=[0]

            if len(cor['non_zero_gbj'])>0:
                gbj_max=(cor['non_zero_gbj']+1)[np.argmax(gbj)]
                precision['gbj']+=[float(len(set(p_val_ind[0:gbj_max])&set(h_ind)))/gbj_max]
                recall['gbj']+=[float(len(set(p_val_ind[0:gbj_max])&set(h_ind)))/eps]
            else:
                precision['gbj']+=[0]
                recall['gbj']+=[0]

            if h_stats['ggnull'][i]>=alpha['ggnull']:
                if len(cor['non_zero_ggnull'])==0:
                    print(h1_stats['ggnull'][i],alpha['ggnull'])
                ggnull_max=(cor['non_zero_ggnull']+1)[np.argmax(ggnull)]
                precision['ggnull']+=[float(len(set(p_val_ind[0:ggnull_max])&set(h_ind)))/ggnull_max]
                recall['ggnull']+=[float(len(set(p_val_ind[0:ggnull_max])&set(h_ind)))/eps]
            else:
                precision['ggnull']+=[0]
                recall['ggnull']+=[0]

            if h_stats['fdr_bh'][i]>=0:
                fdr_bh_max=(cor['non_zero']+1)[max(np.where(fdr_bh>=0)[0])]
                precision['fdr_bh']+=[float(len(set(p_val_ind[0:fdr_bh_max])&set(h_ind)))/fdr_bh_max]
                recall['fdr_bh']+=[float(len(set(p_val_ind[0:fdr_bh_max])&set(h_ind)))/eps]
            else:
                precision['fdr_bh']+=[0]
                recall['fdr_bh']+=[0]

            if len(cor['non_zero_ghc'])>0:
                ghc_max=(cor['non_zero_ghc']+1)[np.argmax(ghc)]
                precision['ghc']+=[float(len(set(p_val_ind[0:ghc_max])&set(h_ind)))/ghc_max]
                recall['ghc']+=[float(len(set(p_val_ind[0:ghc_max])&set(h_ind)))/eps] 
            else:
                precision['ghc']+=[0]
                recall['ghc']+=[0]
    
    if null:
        for stat in Stats:
            alpha[stat]=np.percentile(h_stats[stat],95)
        alpha['fdr_bh']=0

        parms.update(alpha)
        alphaCSV.append(pd.DataFrame(parms,index=[0])).to_csv('alpha.csv',index=False)
    else:
        for stat in Stats+['fdr_bh']:
            power[stat]=np.mean(np.array(h1_stats[stat])>=alpha[stat])
            if len(precision[stat])>0:
                precision[stat]=np.mean(precision[stat])   
                recall[stat]=np.mean(recall[stat])
            else:
                precision.pop(stat)
                recall.pop(stat)

        power=pd.DataFrame(power,index=[0]).T.reset_index()
        power.columns=['stat','value']
        power.insert(1,'type','power')

        precision=pd.DataFrame(precision,index=[0]).T.reset_index()
        precision.columns=['stat','value']
        precision.insert(1,'type','precision')

        recall=pd.DataFrame(recall,index=[0]).T.reset_index()
        recall.columns=['stat','value']
        recall.insert(1,'type','recall')

        power=power.append(precision)
        power=power.append(recall)
        power.index=[0]*len(power)
        power=power.merge(pd.DataFrame(parms,index=[0]),left_index=True,right_index=True)  
        
    return(power)
