import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed
from multiprocessing import cpu_count
from scipy.stats import norm, beta
import pdb
import psutil
import matplotlib.pylab as plt
import time

from statsPython.qq_var import *
from statsPython.mpHelp import *

def binsMinMax(df,B,N):
    bins=np.array(range(int(max(0,df.bins.min()-N)),int(min(B-1,df.bins.max()+N))+1)).reshape(-1,1)
    k=np.array([df.k.iloc[0]]*len(bins)).reshape(-1,1)
    return(pd.DataFrame(np.concatenate([k,bins],axis=1),columns=['k','bins']))

def kMinMax(df):
    return(pd.DataFrame({'min':df.k.min(),'max':df.k.max()},index=[0]))

def setupELL(N,parms):
    ellEps=parms['ellEps']
    ellDelta=parms['ellDelta']
    
    binPower=parms['binPower']
    name=parms['name']
    local=parms['local']
    
    cpu=parms['cpu']

    N=DBRead(name+'process/traitData',parms,toPickle=True).shape[0]
    H0ZPairWiseCors=DBRead(name+'sim/H0ZPairwiseCors',toPickle=True)    
    
    d=int(N*ellDelta)
    
    if DBIsFile(name+'sim/','ELL',parms):
        return()
                
    numBins=int(d*binPower)
    t0=time.time()
    print('start',round((time.time()-t0),2))
    
    bins=np.append(np.append(np.linspace(0,1/numBins,binPower),np.linspace(1/numBins,delta,numBins+1)[1:-1]),
       np.linspace(ellDelta,1,binPower))
    
    kRan=np.array(range(1,d+1))
    minB=np.digitize(beta.ppf(ellEps,kRan,N-kRan),bins).astype(int)-1   
    maxB=np.digitize(beta.ppf(1-ellEps,kRan,N-kRan),bins).astype(int)-1
    maxB[0]=maxB[1]
    
    numBins=max(maxB)+1
    bins=bins[1:numBins+1]
     
    minMaxK=pd.DataFrame({'binEdge':bins})
    minK=pd.DataFrame({'maxB':maxB,'k':range(d)}).groupby(by='maxB').apply(lambda df: pd.DataFrame({'maxB':df.maxB.iloc[0],
        'k':df.k.min()},index=[0])).reset_index(drop=True)
    minMaxK.insert(1,'minK',minK.k.iloc[minK.maxB.searchsorted(range(numBins),side='left')].values)

    maxK=pd.DataFrame({'minB':minB,'k':range(d)}).groupby(by='minB').apply(lambda df: pd.DataFrame({'minB':df.minB.iloc[0],
        'k':df.k.max()},index=[0])).reset_index(drop=True)
    minMaxK.insert(2,'maxK',maxK.k.iloc[maxK.minB.iloc[1:].searchsorted(range(numBins),side='right')].values)
   
    print('ends', psutil.virtual_memory().percent,round((time.time()-t0),2))
          
    end=np.cumsum(maxB-minB+1) 
    start=np.append([0],end[:-1])
    minMaxB=pd.DataFrame({'start':start,'end':end},dtype='int')
    
    rhoBar=getRhoBar(H0ZPairWiseCors)
    var=pd.DataFrame(columns=['binEdge','var'],dtype='float32')
    
    futures=[]
    with ProcessPoolExecutor(cpu) as executor:    
        for i in range(int(np.ceil(numBins/np.ceil(numBins/cpu)))):
            futures.append(executor.submit(vst,bins[i*int(np.ceil(numBins/cpu)):min(
                (i+1)*int(np.ceil(numBins/cpu)),numBins)],N,rhoBar))
        
        for f in as_completed(futures):
            var=var.append(f.result())
    
    var=var.sort_values(by='binEdge').reset_index(drop=True)
    print('ghcDat', psutil.virtual_memory().percent,round((time.time()-t0),2))
    
    minMaxK.insert(minMaxK.shape[1],'var',var['var']) # binEdge,min,max, var
    print('gamma', psutil.virtual_memory().percent,round((time.time()-t0),2))

    ebb=pd.DataFrame(columns=['binEdge','ell'],dtype='float32')
    
    futures=[]
    with ProcessPoolExecutor(cpu) as executor: 
        for i in np.arange(int(np.ceil(numBins/np.ceil(numBins/cpu)))):
            futures.append(executor.submit(setupELLHelp,minMaxK[i*int(np.ceil(numBins/cpu)):min(
                (i+1)*int(np.ceil(numBins/cpu)),numBins)],N))
                                      
        for f in as_completed(futures):
            ebb=ebb.append(f.result())

    print('mp', psutil.virtual_memory().percent,round((time.time()-t0),2))

    ebb=ebb.sort_values(by='binEdge')
    ELL={}
    for k in range(d):
        ELL[k]=ebb.iloc[minMaxB.loc[k,'start']:minMaxB.loc[k,'end']].reset_index(drop=True)
        ELL[k].loc[:,'binEdge']=(ELL[k].loc[:,'binEdge']-ELL[k].loc[:,'binEdge'].astype(int))
        
    print('ebb', psutil.virtual_memory().percent,round((time.time()-t0),2))

    for k in range(d):
        DBWrite(ELL[k],name+'sim/ELL-'+str(k),parms,toPickle=True)  
    
    return()
               
