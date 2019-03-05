import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from ggof import *
from myStats import *
from ghc import *
from ggnull import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def monteCarlo(H,parms,mu,eps,z):
    B,N=z.shape
    Types=parms['Types']
    sig_tri=parms['sig_tri']
    np.random.seed(1)
    freeze_support()

    arr,cr=ggStats(N)

    F_n=np.array([float(j)/N for j in range(1,N+1)])
        
    z=z.copy()
    if mu*eps>0:
        z[:,range(eps)]+=mu
        
    M=multiprocessing.cpu_count()
    #i=0
    #results=mc((N,F_n,arr,cr,z[i*int(np.ceil(H/M)):min((i+1)*int(np.ceil(H/M)),H)],sig_tri,Types))
    #pdb.set_trace()
    print(psutil.virtual_memory().percent)
    
    #results=[mc((N,F_n,arr,cr,z,sig_tri,Types))]
    t=time.time()
    with ProcessPoolExecutor() as executor:    
        results=executor.map(mc, [(N,F_n,arr,cr,z[i*int(np.ceil(B/M)):min((i+1)*int(np.ceil(B/M)),B)],sig_tri,Types) for i in
                                  range(int(M))])
    
    power=pd.DataFrame()
    fail=pd.DataFrame()
    for result in results:
        power=power.append(result[0])
        fail=fail.append(result[1])
    print('mc '+str((time.time()-t)/60))
    print(psutil.virtual_memory().percent)

    t=time.time()
    powerGG,failGG=ggnull(z,parms['sigName'])
    power=power.append(powerGG)
    print('ggnull '+str((time.time()-t)/60))
    print(psutil.virtual_memory().percent)

    t=time.time()
    power=power.append(ghc(z,parms['sigName']))
    print('ghc '+str((time.time()-t)/60))
    print(psutil.virtual_memory().percent)
    #xx=pd.concat([power[power.Type=='ggnull'].Value.reset_index(),power[power.Type=='ggnull1'].Value.reset_index()],axis=1)
    
    t=time.time()
    power=power.append(myStats(z))
    print('myStats '+str((time.time()-t)/60))
    print(psutil.virtual_memory().percent)
    pdb.set_trace()

    Types=power.Type.drop_duplicates()
    Types=Types[~Types.str.contains('1')]
    for typ in Types:
            print(typ,np.percentile(np.abs(power[power.Type==typ].Value.values-power[power.Type==typ+'1'].Value.values),99))
    pdb.set_trace()
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[mu,eps]],columns=['mu','eps'],index=[0]),left_index=True,right_index=True)

    if len(fail)>0:
        fail.index=len(fail)*[0]
        fail=fail.merge(pd.DataFrame([[mu,eps]],columns=['mu','eps'],index=[0]),left_index=True,right_index=True)

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
        non_zero_hc=non_zero[non_zero>=sum(p_val<1.0/len(p_val))]
        
        if len(set(Types)&set(['ggnull','ghc','gbj']))>0:
            cor=ggof(z[i], p_val,sig_tri,arr,cr,Types)
            
        if 'minP' in Types:
            out+=[['minP',np.max(-np.log(p_val[non_zero]))]]

        if 'hc' in Types:
            hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[non_zero_hc]
            out+=[['hc',np.max(hc)]]

        if 'ghc' in Types:
            out+=[['ghc',np.max(cor['ghc']) if len(cor['ghc'])>0 else 0]]
            fail+=[['ghc',1-float(len(cor['ghc']))/len(non_zero_hc)]]
        
        if 'bj' in Types:
            bj=N*(D(F_n[:-1],p_val[:-1]))[non_zero]
            out+=[['bj',np.max(bj)]]
            
        if 'gbj' in Types:
            out+=[['gbj',np.max(cor['gbj']) if len(cor['gbj'])>0 else 0]]
            fail+=[['gbj',1-float(len(cor['gbj']))/len(non_zero)]]

        if 'gnull' in Types:
            gnull=-beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[non_zero]
            out+=[['gnull',np.max(gnull)]]
        
        if 'ggnull' in Types:
            if len(cor['ggnull'])>0:
                out+=[['ggnull',np.max(cor['ggnull'])]]
            fail+=[['ggnull',1-float(len(cor['ggnull']))/len(non_zero)]]
        
        if 'fdr' in Types:
            fdr=(F_n/p_val)[non_zero]
            out+=[['fdr',np.max(fdr)]]

        if 'score' in Types:
            out+=[['score',np.sum(np.sort(-np.abs(z[i]))[non_zero]**2)]]

    return((pd.DataFrame(out,columns=['Type','Value']),pd.DataFrame(fail,columns=['Type','Value'])))