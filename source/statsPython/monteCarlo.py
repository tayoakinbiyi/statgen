import numpy as np
from ELL.util import memory
from scipy.stats import norm
import pdb
import time
from utility import *
import matplotlib.pyplot as plt

def genRef(func,refReps,maxRefReps,vZ,methodName):    
    t0=time.time()
    
    L=makeL(vZ)
    ref=np.ones(refReps)
    numTraits=vZ.shape[1]
    
    t1=time.time()
    
    mins=0
    length=0
    for block in np.arange(int(np.ceil(refReps/maxRefReps))):
        repRange=np.arange(block*maxRefReps,min(refReps,(block+1)*maxRefReps)).astype(int)
        ans,t_mins=func(-np.sort(-np.abs(np.matmul(norm.rvs(size=[len(repRange),numTraits]),L.T))))
        ref[repRange]=ans
        mins+=t_mins
            
    log('mc genRef {} : {} min'.format(methodName,(t1-t0)/60+mins))
    
    return(ref)

def mcPVal(test,ref):
    t0=time.time()
    
    refReps=len(ref)

    mc=np.zeros(len(test))
    sortOrd=np.argsort(test)
    mc[sortOrd]=(1+np.searchsorted(np.sort(ref),test[sortOrd],side='left'))/(refReps+1)
        
    t1=time.time()
    
    return(mc,(t1-t0)/60)

def mc(func,vZ,ref,methodName,wald):
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)  
    axs.hist(ref[ref<=np.quantile(ref,.0011)],bins=150)
    axs.axvline(x=np.quantile(ref,.0011))
    fig.savefig('diagnostics/genRef<.001-'+methodName+'.png')

    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)  
    axs.hist(ref[ref<=np.quantile(ref,.011)],bins=150)
    axs.axvline(x=np.quantile(ref,.01))
    axs.axvline(x=np.quantile(ref,.0011))
    fig.savefig('diagnostics/genRef<.01-'+methodName+'.png')

    pvals=[]
    test,mins=func(wald)
    log('mc score {}, min {}'.format(methodName,mins))

    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)  
    axs.hist(test[test<np.quantile(ref,.001)],bins=30)
    fig.savefig('diagnostics/'+methodName+'<.001.png')

    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)  
    axs.hist(test[test<=np.quantile(ref,.01)],bins=30)
    fig.savefig('diagnostics/'+methodName+'<.01.png')

    pval,mins=mcPVal(test,ref)
    log('mc pval {}, min {}'.format(methodName,mins))
    
    return(pval.reshape(-1,1))
 