import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil

from ggof import *
from mymath import *

def monteCarlo(dat):
    j=0
    H=dat[j];j+=1
    N=dat[j];j+=1
    mu=dat[j];j+=1
    eps=dat[j];j+=1
    sigName=dat[j];j+=1
    L=dat[j];j+=1
    Types=dat[j];j+=1
    alpha=dat[j];j+=1

    np.random.seed(1)

    arr,cr=ggStats(N)

    F_n=np.array([float(j)/N for j in range(1,N+1)])

    if L is None:
        z=np.random.normal(0,1,size=(H,N))
        sig_tri=np.array([0]*int(N*(N-1)/2))
    else:
        z=np.matmul(L.T,np.random.normal(0,1,size=(N,H))).T
        sig_tri=np.matmul(L.T,L)[np.triu_indices(N,1)].flatten()
        
    if mu*eps>0:
        z[:,range(eps)]+=mu
               
    p=2*norm.cdf(-np.abs(z))  
    
    power=[]
    fail=[]

    for i in range(len(p)):   
        if i%1000==0:
            print(N,mu,eps,sigName,str(int(100*i/len(p)))+'%')
           
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
        non_zero = np.array(range(int(np.ceil(len(p_val)/2.0))))
        non_zero_hc=non_zero[non_zero>=sum(p_val<=1.0/len(p_val))]
        non_zero=non_zero.tolist()
        non_zero_hc=non_zero_hc.tolist()
        
        if len(set(Types)&set(['ggnull','ghc','gbj']))>0:
            cor=ggof(z[i], p_val,sig_tri,arr,cr,Types)

        if len([x for x in Types if x=='minP']):
            power+=[['minP',np.max(-np.log(p_val[non_zero]))]]

        if len([x for x in Types if x=='hc']):
            hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[non_zero_hc]
            power+=[['hc',np.max(hc)]]

        if len([x for x in Types if x=='ghc']):
            power+=[['ghc',np.max(cor['ghc']) if len(cor['ghc'])>0 else 0]]
            fail+=[['ghc',1-float(len(cor['ghc']))/len(non_zero_hc)]]
        
        if len([x for x in Types if x=='bj']):
            bj=N*(D(F_n[:-1],p_val[:-1]))[non_zero]
            power+=[['bj',np.max(bj)]]
            
        if len([x for x in Types if x=='gbj']):
            power+=[['gbj',np.max(cor['gbj']) if len(cor['gbj'])>0 else 0]]
            fail+=[['gbj',1-float(len(cor['gbj']))/len(non_zero)]]

        if len([x for x in Types if x=='gnull']):
            gnull=-beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[non_zero]
            power+=[['gnull',np.max(gnull)]]
        
        if len([x for x in Types if x=='ggnull']):
            if len(cor['ggnull'])>0:
                power+=[['ggnull',np.max(cor['ggnull'])]]
            fail+=[['ggnull',1-float(len(cor['ggnull']))/len(non_zero)]]
        
        if len([x for x in Types if x=='fdr']):
            fdr=(F_n/p_val)[non_zero]
            power+=[['fdr',np.max(fdr)]]

        if len([x for x in Types if x=='score']):
            power+=[['score',np.sum(z[i]**2)]]

    power=pd.DataFrame(power,columns=['Type','Value'])
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[mu,eps]],columns=['mu','eps'],index=[0]),left_index=True,right_index=True)

    fail=pd.DataFrame(fail,columns=['Type','Value'])
    fail.index=len(fail)*[0]
    fail=fail.merge(pd.DataFrame([[mu,eps]],columns=['mu','eps'],index=[0]),left_index=True,right_index=True)
            
    return({'power':power,'fail':fail,'alpha':alpha})        

