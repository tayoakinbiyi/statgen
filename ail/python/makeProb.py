import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy.stats import norm, beta
import pdb
import psutil
import matplotlib.pylab as plt
import time

from python.qq_var import *
from python.ggof import *
from python.mpHelp import *

def binsMinMax(df,B,N):
    bins=np.array(range(int(max(0,df.bins.min()-N)),int(min(B-1,df.bins.max()+N))+1)).reshape(-1,1)
    k=np.array([df.k.iloc[0]]*len(bins)).reshape(-1,1)
    return(pd.DataFrame(np.concatenate([k,bins],axis=1),columns=['k','bins']))

def kMinMax(df):
    return(pd.DataFrame({'min':df.k.min(),'max':df.k.max()},index=[0]))

def makeProb(L,parms):
    eps=1e-10
    binPower=500
    
    N=parms['N']
    sigName=parms['sigName']
    
    d=int(N/2)

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(7,forward=True)
    off_diag=parms['sig'][np.triu_indices(N,1)].flatten()  
    axs.hist(off_diag,bins=np.linspace(-1,1,100))
    fig.suptitle(sigName)
    fig.savefig('../sig_hist/'+sigName+'.png',bbox_inches='tight')
    plt.close()    

    z=np.matmul(L.T,np.random.normal(0,1,size=(N,200))).T
    pairwise_cors=np.corrcoef(z,rowvar=False)[np.triu_indices(N,1)].flatten()#np.array([0]*int(N*(N-1)/2))
    mkdir=False
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(7,forward=True)
    axs.hist(pairwise_cors,bins=np.linspace(-1,1,100))
    fig.suptitle(sigName)
    fig.savefig('../sig_hist/'+sigName+'_est.png',bbox_inches='tight')
    plt.close()    

    if os.path.isdir('../ebb/'+sigName):
        ggnullDat={}
        for k in range(d):
            ggnullDat[k]=pd.read_csv('../ebb/'+sigName+'/ggnullDat-'+str(k)+'.csv',dtype='float32')
        ghcDat=pd.read_csv('../ebb/'+sigName+'/ghcDat.csv',dtype='float32')
        return(ggnullDat,ghcDat)
    else:
        mkdir=True
            
    M=multiprocessing.cpu_count()
    
    numBins=int(N*binPower)
    t0=time.time()
    print('start',round((time.time()-t0),2))
    
    bins=np.append(np.append(np.linspace(0,1/numBins,binPower),np.linspace(1/numBins,.5,numBins+1)[1:-1]),np.linspace(.5,1,binPower))
    
    kRan=np.array(range(1,d+1))
    minB=np.digitize(beta.ppf(eps,kRan,N-kRan),bins).astype(int)-1   
    maxB=np.digitize(beta.ppf(1-eps,kRan,N-kRan),bins).astype(int)-1
    maxB[0]=maxB[1]
    pdb.set_trace()
    
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
    
    rhoBar=getRhoBar(pairwise_cors)
    futures=[]
    with ProcessPoolExecutor() as executor:    
        for i in range(int(np.ceil(numBins/np.ceil(numBins/M)))):
            futures.append(executor.submit(vst,bins[i*int(np.ceil(numBins/M)):min((i+1)*int(np.ceil(numBins/M)),numBins)],N,rhoBar))
        
    ghcDat=pd.DataFrame(columns=['binEdge','var'],dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        ghcDat=ghcDat.append(f.result())
    
    ghcDat=ghcDat.sort_values(by='binEdge').reset_index(drop=True)
    print('ghcDat', psutil.virtual_memory().percent,round((time.time()-t0),2))

    minMaxK.insert(minMaxK.shape[1],'var',ghcDat['var']) # binEdge,min,max, var
    print('gamma', psutil.virtual_memory().percent,round((time.time()-t0),2))

    futures=[]
    mpHelp(minMaxK,N)
    with ProcessPoolExecutor() as executor: 
        for i in range(int(np.ceil(numBins/np.ceil(numBins/M)))):
            futures.append(executor.submit(mpHelp,minMaxK[i*int(np.ceil(numBins/M)):min((i+1)*int(np.ceil(numBins/M)),numBins)],N))
                                      
    ebb=pd.DataFrame(columns=['binEdge','ggnull'],dtype='float32')
    for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
        ebb=ebb.append(f.result())

    print('mp', psutil.virtual_memory().percent,round((time.time()-t0),2))

    ebb=ebb.sort_values(by='binEdge')
    ggnullDat={}
    for k in range(d):
        ggnullDat[k]=ebb.iloc[minMaxB.loc[k,'start']:minMaxB.loc[k,'end']].reset_index(drop=True)
        ggnullDat[k].loc[:,'binEdge']=(ggnullDat[k].loc[:,'binEdge']-ggnullDat[k].loc[:,'binEdge'].astype(int))
        
    print('ebb', psutil.virtual_memory().percent,round((time.time()-t0),2))

    os.mkdir('../ebb/'+sigName)
    pd.Series(rhoBar,name='rhoBar').to_csv('../ebb/'+sigName+'/rhoBar.csv',index=False,header=True)
    for k in range(d):
        ggnullDat[k].to_csv('../ebb/'+sigName+'/ggnullDat-'+str(k)+'.csv',index=False)  
    np.savetxt('../ebb/'+sigName+'/pairwise_cors.csv',pairwise_cors,delimiter=',')
    ghcDat.to_csv('../ebb/'+sigName+'/ghcDat.csv',index=False)
    
    return(ggnullDat,ghcDat)
           
def vst(binEdges,d,rhoBar):
    z=np.abs(norm.ppf(binEdges/2))
    return(pd.DataFrame({'binEdge':binEdges,'var':getVarNoMu(z,d,rhoBar)},dtype='float32'))                     
    