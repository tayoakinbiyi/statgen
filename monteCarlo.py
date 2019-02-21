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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def monteCarlo(H,parms,mu,eps,z):
    N=parms['N']
    Types=parms['Types']
    sig_tri=parms['sig_tri']
    np.random.seed(1)
    freeze_support()

    arr,cr=ggStats(N)

    F_n=np.array([float(j)/N for j in range(1,N+1)])
        
    if True:#mu*eps>0:
        z[:,range(10)]+=3
        
    M=float(cpu_count())
    i=0
    results=mc((N,F_n,arr,cr,z[i*int(np.ceil(H/M)):min((i+1)*int(np.ceil(H/M)),H)],sig_tri,Types))
    pdb.set_trace()
    print(mu,eps,psutil.virtual_memory().percent)
        
    with ProcessPoolExecutor() as executor:    
        results=executor.map(mc, [(N,F_n,arr,cr,z[i*int(np.ceil(H/M)):min((i+1)*int(np.ceil(H/M)),H)],sig_tri,Types) for i in
                                  range(int(M))])
    
    power=pd.DataFrame()
    fail=pd.DataFrame()
    for result in results:
        power=power.append(result[0])
        fail=fail.append(result[1])
    
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[mu,eps]],columns=['mu','eps'],index=[0]),left_index=True,right_index=True)

    if len(fail)>0:
        fail.index=len(fail)*[0]
        fail=fail.merge(pd.DataFrame([[mu,eps]],columns=['mu','eps'],index=[0]),left_index=True,right_index=True)

    if mu*eps>0:
        z[:,range(eps)]-=mu

    return((power,fail))

def mc(data):
    j=0
    N=data[j];j+=1
    F_n=data[j];j+=1
    arr=data[j];j+=1
    cr=data[j];j+=1
    z=data[j];j+=1
    sig_tri=data[j];j+=1
    Types=data[j];j+=1
    
    p=2*norm.cdf(-np.abs(z))  
    
    out=[]
    fail=[]

    for i in range(len(p)):   
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
  
        non_zero = np.array(range(int(np.ceil(len(p_val)/2.0))))
        non_zero_hc=non_zero[non_zero>=sum(p_val<=1.0/len(p_val))]
        #non_zero=non_zero.tolist()
        #non_zero_hc=non_zero_hc.tolist()
        
        if len(set(Types)&set(['ggnull','ghc','gbj']))>0:
            cor=ggof(z[i], p_val,sig_tri,arr,cr,Types)

        if len([x for x in Types if x=='minP']):
            out+=[['minP',np.max(-np.log(p_val[non_zero]))]]

        if len([x for x in Types if x=='hc']):
            hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[non_zero_hc]
            out+=[['hc',np.max(hc)]]

        if len([x for x in Types if x=='ghc']):
            out+=[['ghc',np.max(cor['ghc']) if len(cor['ghc'])>0 else 0]]
            fail+=[['ghc',1-float(len(cor['ghc']))/len(non_zero_hc)]]
        
        if len([x for x in Types if x=='bj']):
            bj=N*(D(F_n[:-1],p_val[:-1]))[non_zero]
            out+=[['bj',np.max(bj)]]
            
        if len([x for x in Types if x=='gbj']):
            out+=[['gbj',np.max(cor['gbj']) if len(cor['gbj'])>0 else 0]]
            fail+=[['gbj',1-float(len(cor['gbj']))/len(non_zero)]]

        if len([x for x in Types if x=='gnull']):
            gnull=-beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[non_zero]
            out+=[['gnull',np.max(gnull)]]
        
        if len([x for x in Types if x=='ggnull']):
            if len(cor['ggnull'])>0:
                out+=[['ggnull',np.max(cor['ggnull'])]]
            fail+=[['ggnull',1-float(len(cor['ggnull']))/len(non_zero)]]
        
        if len([x for x in Types if x=='fdr']):
            fdr=(F_n/p_val)[non_zero]
            out+=[['fdr',np.max(fdr)]]

        if len([x for x in Types if x=='score']):
            out+=[['score',np.sum(z[i]**2)]]

    return((pd.DataFrame(out,columns=['Type','Value']),pd.DataFrame(fail,columns=['Type','Value'])))