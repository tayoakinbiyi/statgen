import numpy as np
import pdb
import pandas as pd
from scipy.stats import norm,beta
import time
from myMath import *
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
from concurrent.futures import ProcessPoolExecutor
from qq_var import *
import random
import os
    
def ebb_gnull(x,ar,cr):
    k=int(x[0])
    lam=x[1]
    gamma=x[2]
    return(1-sum(np.exp(cr[0:(k+1)]+np.nansum(np.log(ar[0:(k+1)]*gamma+1-lam),axis=1)+
        np.nansum(np.log(ar.T[0:(k+1)]*gamma+lam),axis=1)-np.nansum(np.log(1+ar[0]*gamma)))))

def ggof(z_raw,p,pairwise_cors,arr,cr):
    z = -np.sort(-np.abs(z_raw))
    cor={}
                             
    d=len(z)
    k_vec=np.array(range(d))

    non_zero = np.array(range(int(np.ceil(d/2.0))))
    cor['non_zero']=non_zero    
    cor['non_zero_hc']=non_zero[non_zero>=sum(p<=1.0/d)]

    sigmasq = np.array(range(d),dtype='float')    
    sigmasq[non_zero]=var_st(z[non_zero],d,pairwise_cors)
    
    lam = np.array(range(d),dtype='float')
    rho = np.array(range(d),dtype='float')
    gamma =np.array(range(d),dtype='float')

    lam[non_zero] = 2*norm.sf(z[non_zero])
    rho[non_zero] = (sigmasq[non_zero] - d*lam[non_zero]*(1-lam[non_zero])) / (d*(d-1)*lam[non_zero]*(1-lam[non_zero]))
    gamma[non_zero] = rho[non_zero] / (1-rho[non_zero])
    
    pq_mat = np.array([[0,0]]*d,dtype='float')
    pq_mat[non_zero, 0] = -lam[non_zero] / (d-1)
    pq_mat[non_zero, 1] = -(1-lam[non_zero]) / (d-1)
    gamma_check = pq_mat.max(1)   

    non_zero = non_zero[gamma[non_zero] >= gamma_check[non_zero]]
    cor['non_zero_ggnull']=non_zero

    if len(non_zero)==0:
        return ({'non_zero':cor['non_zero'],'non_zero_hc':cor['non_zero_hc'],'non_zero_ghc':np.array([]),
                 'non_zero_gbj':np.array([]),'non_zero_ggnull':np.array([]),'ghc':[],'gbj':[],'ggnull':[]})

    param = np.concatenate([k_vec[non_zero].reshape(-1,1),lam[non_zero].reshape(-1,1),gamma[non_zero].reshape(-1,1)],axis=1)
    cor['ggnull'] = -np.apply_along_axis(ebb_gnull,1, param,ar=arr,cr=cr)
                             
    non_zero_ghc=non_zero[non_zero>=sum(p<1.0/d)]
    cor['non_zero_ghc']=non_zero_ghc
    cor['ghc']=(k_vec[non_zero_ghc]+1-d*p[non_zero_ghc])/np.sqrt(sigmasq[non_zero_ghc])
    cor['var']=sigmasq[non_zero_ghc]
    cor['k']=k_vec[non_zero_ghc]
    
    if 'gbj' in ['5']:#Types:
        muj =np.array([0]*d,dtype='float')
        non_zero_gbj=non_zero[p[non_zero]<(k_vec[non_zero]+1)/float(d)]
        for iii in non_zero_gbj:
            muj[iii] = scipy.optimize.brentq(qnorm_mu, 0, 1000, args = (z[iii],k_vec[iii]+1, d))

        sigmasq_alt = np.array(range(d),dtype='float')   
        sigmasq_alt[non_zero_gbj]=var_st_mu(z[non_zero_gbj], d,mu=muj[non_zero_gbj],pairwise_cors=pairwise_cors)

        lam_alt=np.array([0]*d,dtype='float')
        rho_alt = np.array([0]*d,dtype='float')
        gamma_alt = np.array([0]*d,dtype='float')

        lam_alt[non_zero_gbj] = (k_vec[non_zero_gbj]+1) / float(d)
        rho_alt[non_zero_gbj] = (sigmasq_alt[non_zero_gbj] - d*lam_alt[non_zero_gbj]*(1-lam_alt[non_zero_gbj])) / \
             (d*(d-1)*lam_alt[non_zero_gbj]*(1-lam_alt[non_zero_gbj]))
        gamma_alt[non_zero_gbj] = rho_alt[non_zero_gbj] / (1-rho_alt[non_zero_gbj])

        pq_mat_alt =np.array([[0,0]]*d,dtype='float')
        pq_mat_alt[non_zero_gbj, 0] = -lam_alt[non_zero_gbj] / (d-1)
        pq_mat_alt[non_zero_gbj, 1] = -(1-lam_alt[non_zero_gbj]) / (d-1)    
        gamma_check_alt = pq_mat_alt.max(1)

        non_zero_gbj = non_zero_gbj[gamma_alt[non_zero_gbj] >= gamma_check_alt[non_zero_gbj]] 

        cor['non_zero_gbj']=non_zero_gbj
        if len(non_zero_gbj)==0:
            cor['gbj']=[]
            return(cor)

        null_loglik = np.array([0]*d,dtype='float')
        param = np.concatenate([k_vec[non_zero_gbj].reshape(-1,1),lam[non_zero_gbj].reshape(-1,1),
                                gamma[non_zero_gbj].reshape(-1,1)],axis=1)
        null_loglik[non_zero_gbj] = np.apply_along_axis(ebb_loglik,1,param,d=d,ar=arr,cr=cr)

        alt_loglik = np.array([0]*d,dtype='float')
        alt_param = np.concatenate([k_vec[non_zero_gbj].reshape(-1,1),lam_alt[non_zero_gbj].reshape(-1,1),
            gamma_alt[non_zero_gbj].reshape(-1,1)],axis=1)

        alt_loglik[non_zero_gbj] = np.apply_along_axis(ebb_loglik,1,alt_param,d=d,ar=arr,cr=cr)
        cor['gbj']= alt_loglik[non_zero_gbj] - null_loglik[non_zero_gbj]

    return(cor)

def ebb_loglik(x, d,ar,cr):
    k=int(x[0])+1
    lam=x[1]
    gamma=x[2]
    ans=cr[k] + np.nansum(np.log(lam+gamma*ar.T[k])) + np.nansum(np.log(1-lam+gamma*ar[k]))- np.nansum(np.log(1+ar[0]*gamma))
    return(ans)
