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

def monteCarlo(L,sigName,eps,mu,Reps,ebb,var):
    pairwise_cors=np.loadtxt('ebb/'+sigName+'/pairwise_cors.csv')
    N=len(L)
    
    z=np.matmul(L.T,np.random.normal(0,1,size=(N,Reps))).T
    
    if mu*eps>0:
        z[:,range(eps)]+=mu
    
    power=pd.DataFrame()
    fail=pd.DataFrame()
    
    print(eps,mu,psutil.virtual_memory().percent)
    powerGG,failGG=ggnull(z,sigName,ebb)
    print('ggnull',psutil.virtual_memory().percent)
    power=power.append(powerGG)
    fail=fail.append(failGG)
    power=power.append(ghc(z,sigName,var))
    print('ghc',psutil.virtual_memory().percent)
    power=power.append(myStats(z))
    print('myStats',psutil.virtual_memory().percent)
    
    M=multiprocessing.cpu_count()
    with ProcessPoolExecutor() as executor:    
        results=executor.map(mc, [(z[i*int(np.ceil(Reps/M)):min((i+1)*int(np.ceil(Reps/M)),Reps)].tolist(),pairwise_cors) for
            i in range(int(np.ceil(Reps/np.ceil(Reps/M))))])
    
    for res in results:
        power=power.append(res[0])
        fail=fail.append(res[1])
    
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[eps,mu]],columns=['eps','mu'],index=[0]),left_index=True,right_index=True)

    if len(fail)>0:
        fail.index=len(fail)*[0]
        fail=fail.merge(pd.DataFrame([[eps,mu]],columns=['eps','mu'],index=[0]),left_index=True,right_index=True)

    return(power,fail)

def mc(data):
    j=0
    z=np.array(data[j]);j+=1
    pairwise_cors=data[j];j+=1
    
    Reps,N=z.shape
    arr,cr=ggStats(N)

    F_n=np.array([float(j)/N for j in range(1,N+1)])

    p=2*norm.cdf(-np.abs(z))  
    
    out=[]
    val=[]
    fail=[]

    for i in range(len(p)):   
        p_val_ind=np.argsort(p[i])
        p_val=p[i][p_val_ind]
  
        non_zero = np.array(range(int(np.ceil(len(p_val)/2.0))))
        non_zero_hc=non_zero[non_zero>=sum(p_val<1.0/N)]
        
        cor=ggof(z[i], p_val,pairwise_cors,arr,cr)
            
        out+=[['minP0',np.max(-np.log(p_val[non_zero]))]]

        hc=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[non_zero_hc]
        out+=[['hc0',np.max(hc)]]

        hcFull=(np.sqrt(N)*((F_n-p_val)/np.sqrt(p_val*(1-p_val))))[non_zero]
        out+=[['hcFull0',np.max(hcFull)]]

        out+=[['ghc0',np.max(cor['ghc']) if len(cor['ghc'])>0 else 0]]
        fail+=[['ghc0',1-float(len(cor['ghc']))/len(non_zero_hc)]]
        
        bj=N*(D(F_n[:-1],p_val[:-1]))[non_zero]
        out+=[['bj0',np.max(bj)]]
            
        gnull=-beta.cdf(p_val,range(1,N+1), [N + 1 - j for j in range(1,N+1)])[non_zero]
        out+=[['gnull0',np.max(gnull)]]
        
        if len(cor['ggnull'])>0:
            out+=[['ggnull0',max(cor['ggnull'])]]
            #out+=[['ggnull00',cor['ggnull'][x],p_val[non_zero][x],x] for x in range(len(cor['ggnull']))]
        fail+=[['ggnull0',1-float(len(cor['ggnull']))/len(non_zero)]]
        
        fdr=(F_n/p_val)[non_zero]
        out+=[['fdr0',np.max(fdr)]]

        out+=[['score0',np.sum(np.sort(-np.abs(z[i]))[non_zero]**2)]]
        
    return(pd.DataFrame(out,columns=['Type','Value']),pd.DataFrame(fail,columns=['Type','Value']))