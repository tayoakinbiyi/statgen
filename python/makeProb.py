import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy.stats import norm, beta
from qq_var import *
import pdb
import psutil
import matplotlib.pylab as plt
import time
from ggof import *
from mpHelp import *
#from apply import *

def binsMinMax(df,B,N):
    bins=np.array(range(int(max(0,df.bins.min()-N)),int(min(B-1,df.bins.max()+N))+1)).reshape(-1,1)
    k=np.array([df.k.iloc[0]]*len(bins)).reshape(-1,1)
    return(pd.DataFrame(np.concatenate([k,bins],axis=1),columns=['k','bins']))

def kMinMax(df):
    return(pd.DataFrame({'min':df.k.min(),'max':df.k.max()},index=[0]))

def makeProb(L,parms):
    eps=1e-8
    binPower=700
    
    N=parms['N']
    sigName=parms['sigName']
    
    d=int(N/2)

    z=np.matmul(L.T,np.random.normal(0,1,size=(N,200))).T
    pairwise_cors=np.corrcoef(z,rowvar=False)[np.triu_indices(N,1)].flatten()   

    if os.path.isfile('ebb/'+sigName+'/ebb.csv'):
        ebb=pd.read_csv('ebb/'+sigName+'/ebb.csv',dtype='float32')
        var=pd.read_csv('ebb/'+sigName+'/var.csv',dtype='float32')
        return(ebb,var)
    else:
        try:
            os.mkdir('ebb/'+sigName)
        except:
            print('directory already exists')
    
    M=multiprocessing.cpu_count()
    
    numBins=int(N*binPower)
    t0=time.time()
    print('start',round((time.time()-t0),2))
    
    bins=np.append(np.linspace(0,.5,numBins+1)[:-1],np.linspace(.5,1,binPower))
    
    kRan=np.array(range(1,d+1))
    minB=np.digitize(beta.ppf(eps,kRan,N-kRan),bins).astype(int)-1   
    maxB=np.digitize(beta.ppf(1-eps,kRan,N-kRan),bins).astype(int)-1
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
    minMaxB=pd.DataFrame({'start':start,'end':end},dtype='float32')
    
    rhoBar=getRhoBar(pairwise_cors)
    futures=[]
    with ProcessPoolExecutor() as executor:    
        for i in range(int(np.ceil(numBins/np.ceil(numBins/M)))):
            futures.append(executor.submit(vst,bins[i*int(np.ceil(numBins/M)):min((i+1)*int(np.ceil(numBins/M)),numBins)],N,rhoBar))
        
    var=pd.DataFrame(columns=['binEdge','var'],dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        var=var.append(f.result())
    
    var=var.sort_values(by='binEdge').reset_index(drop=True)
    print('vst', psutil.virtual_memory().percent,round((time.time()-t0),2))

    minMaxK.insert(minMaxK.shape[1],'var',var['var']) # binEdge,min,max, var
    print('gamma', psutil.virtual_memory().percent,round((time.time()-t0),2))

    futures=[]
    #i=0
    #pdb.set_trace()
    #mpHelp(minMaxK.iloc[27726-5:27726+5],N)
    with ProcessPoolExecutor() as executor: 
        for i in range(int(np.ceil(numBins/np.ceil(numBins/M)))):
            futures.append(executor.submit(mpHelp,minMaxK[i*int(np.ceil(numBins/M)):min((i+1)*int(np.ceil(numBins/M)),numBins)],N))
                                      
    ebb=pd.DataFrame(columns=['binEdge','ebb'],dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        ebb=ebb.append(f.result())

    print('mp', psutil.virtual_memory().percent,round((time.time()-t0),2))

    ebb=ebb.sort_values(by='binEdge')
    ebb['binEdge']=(ebb['binEdge']-ebb['binEdge'].astype(int))
    print('ebb', psutil.virtual_memory().percent,round((time.time()-t0),2))

    minMaxB.to_csv('ebb/'+sigName+'/minMaxB.csv',index=False)
    pd.Series(rhoBar,name='rhoBar').to_csv('ebb/'+sigName+'/rhoBar.csv',index=False,header=True)
    ebb.to_csv('ebb/'+sigName+'/ebb.csv',index=False)  
    np.savetxt('ebb/'+sigName+'/pairwise_cors.csv',pairwise_cors,delimiter=',')
    var.to_csv('ebb/'+sigName+'/var.csv',index=False)
    
    return(ebb,var)
           
def vst(binEdges,d,rhoBar):
    z=np.abs(norm.ppf(binEdges/2))
    return(pd.DataFrame({'binEdge':binEdges,'var':getVarNoMu(z,d,rhoBar)},dtype='float32'))                     
    
