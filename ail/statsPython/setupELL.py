import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import cpu_count
from scipy.stats import norm, beta
import pdb
import psutil
import matplotlib.pylab as plt
import time

from ail.statsPython.qq_var import *
from ail.statsPython.setupELLHelp import *
from ail.opPython.DB import *

def binsMinMax(df,B,N):
    bins=np.array(range(int(max(0,df.bins.min()-N)),int(min(B-1,df.bins.max()+N))+1)).reshape(-1,1)
    k=np.array([df.k.iloc[0]]*len(bins)).reshape(-1,1)
    return(pd.DataFrame(np.concatenate([k,bins],axis=1),columns=['k','bins']))

def kMinMax(df):
    return(pd.DataFrame({'min':df.k.min(),'max':df.k.max()},index=[0]))

def setupELL(parms):
    name=parms['name']

    DBCreateFolder(name,'ELL',parms)

    snps=pd.Series(DBListFolder(name+'offDiag',parms),name='traits')
    snps=snps[snps.str.contains('offDiag-')].str.slice(8).values.tolist()
    
    DBSyncLocal(name+'process',parms)
    
    for snp in snps:
        setupELLSnp(snp,parms)
    
def setupELLSnp(snp,parms):
    ellDSet=parms['ellDSet']
    ellEps=parms['ellEps']
    
    binsPerIndex=parms['binsPerIndex']
    name=parms['name']
    local=parms['local']
    numCores=parms['numCores']

    traitData=DBRead(name+'process/traitData',parms,True)

    N=sum(traitData['chr']!=snp)
    d=int(N*max(ellDSet))

    print('loading offDiag '+snp,flush=True)
    H0ZPairWiseCors=DBRead(name+'offDiag/offDiag-'+snp,parms,True)    
        
    numBins=int(d*binsPerIndex)
    t0=time.time()
    print('bins '+snp,round((time.time()-t0),2))
    
    bins=np.append(np.append(np.linspace(0,1/numBins,binsPerIndex),np.linspace(1/numBins,.5,numBins+1)[1:-1]),
       np.linspace(.5,1,binsPerIndex))
    
    kRan=np.arange(1,d+1)
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
   
    print('minMaxK '+snp, psutil.virtual_memory().percent,round((time.time()-t0),2))
          
    end=np.cumsum(maxB-minB+1) 
    start=np.append([0],end[:-1])
    minMaxB=pd.DataFrame({'start':start,'end':end},dtype='int')
    
    rhoBar=getRhoBar(H0ZPairWiseCors)
    var=pd.DataFrame(columns=['binEdge','var'],dtype='float32')
    
    futures=[]
    with ProcessPoolExecutor(numCores) as executor:    
        for i in range(int(np.ceil(numBins/np.ceil(numBins/numCores)))):
            futures.append(executor.submit(vst,bins[i*int(np.ceil(numBins/numCores)):min(
                (i+1)*int(np.ceil(numBins/numCores)),numBins)],N,rhoBar))
        
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            var=var.append(f.result())
    
    var=var.sort_values(by='binEdge').reset_index(drop=True)   
    minMaxK.insert(minMaxK.shape[1],'var',var['var']) # binEdge,min,max, var
    print('gamma '+snp, psutil.virtual_memory().percent,round((time.time()-t0),2))

    ebb=pd.DataFrame(columns=['binEdge','ell'],dtype='float32')
    
    futures=[]
    with ProcessPoolExecutor(numCores) as executor: 
        for i in np.arange(int(np.ceil(numBins/np.ceil(numBins/numCores)))):
            futures.append(executor.submit(setupELLHelp,minMaxK[i*int(np.ceil(numBins/numCores)):min(
                (i+1)*int(np.ceil(numBins/numCores)),numBins)],N))
                                      
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            ebb=ebb.append(f.result())

    print('mp '+snp, psutil.virtual_memory().percent,round((time.time()-t0),2))

    ebb=ebb.sort_values(by='binEdge')
    ELLAll={}
    for k in range(d):
        ELL=ebb.iloc[minMaxB.loc[k,'start']:minMaxB.loc[k,'end']].reset_index(drop=True)
        ELL.loc[:,'binEdge']=(ELL.loc[:,'binEdge']-ELL.loc[:,'binEdge'].astype(int))
        ELLAll[k]=ELL
        print('setupELL '+snp+' '+str(k+1)+' of '+str(d),flush=True)
    
    DBWrite(ELLAll,name+'ELL/ELL-'+snp,parms,toPickle=True)  
    print('finished setupELL '+snp,flush=True)
    
    return()
               
