import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm, beta
from qq_var import *
import pdb
import psutil
import matplotlib.pylab as plt
#from apply import *

def newC(n):
    lFac=np.log(list(range(1,n+1)))
    forw=np.append([0],np.cumsum(lFac))
    bacw=np.append([0],np.cumsum(lFac[::-1]))
    return(bacw[0:int(n/2)]-forw[0:int(n/2)])
    
def binsMinMax(df,B,N):
    bins=np.array(range(int(max(0,df.bins.min()-N)),int(min(B-1,df.bins.max()+N))+1)).reshape(-1,1)
    k=np.array([df.k.iloc[0]]*len(bins)).reshape(-1,1)
    return(pd.DataFrame(np.concatenate([k,bins],axis=1),columns=['k','bins']))

def kMinMax(df):
    return(pd.DataFrame({'min':df.k.min(),'max':df.k.max()},index=[0]))

def makeZ(dat):
    j=0
    z=np.array(dat[j]);j+=1
    mu=dat[j];j+=1
    eps=dat[j];j+=1
    bins=dat[j];j+=1
    
    Reps,N=z.shape
    d=int(N/2)
    
    if mu*eps>0:
        z[:,range(eps)]+=mu

    res=np.concatenate([np.digitize(np.sort(2*norm.sf(np.abs(z)))[:,0:d].reshape(-1,1),bins)-1,[[x] for x in range(d)]*len(z)],axis=1)
    res=pd.DataFrame(res,columns=['bins','k']).groupby('k').apply(binsMinMax,B=len(bins)-1,N=N).reset_index(drop=True)
    return(res)

def makeProb(L,parms):
    eps=1e-8
    binPower=1.5
    
    N=parms['N']
    sigName=parms['sigName']
    
    d=int(N/2)

    z=np.matmul(L.T,np.random.normal(0,1,size=(N,200))).T
    pairwise_cors=np.array([0]*int((N-1)*N/2)) #np.corrcoef(z,rowvar=False)[np.triu_indices(N,1)].flatten()   

    if os.path.isfile(sigName+'-'+str(N)+'-ebb-prob.csv'):
        return()
    
    M=multiprocessing.cpu_count()
    
    numBins=int(N**binPower)
    
    bins=np.linspace(0,.5,numBins+1)
    bins=np.concatenate([np.linspace(0,bins[1],1000),bins[2:-1],np.linspace(bins[-1],1,N)]).flatten()
    
    kRan=np.array(range(1,d+1))
    minB=np.digitize(beta.ppf(eps,kRan,N-kRan),bins).astype(int)-1   
    maxB=np.digitize(beta.ppf(1-eps,kRan,N-kRan),bins).astype(int)-1
    maxB[0]=maxB[1]
    
    numBins=max(maxB)+1
    bins=bins[:numBins+1]
    mids=(bins[1:]+bins[:-1])/2
    
    bins[-1]=1 # make sure any lam is mathematically ok
    mids[-1]=1-1e-8 # conservative last mid for rare very large observed lam
    
    res=pd.DataFrame({'mids':mids,'binEdge':bins[1:]})
    minK=pd.DataFrame({'maxB':maxB,'k':range(d)}).groupby(by='maxB').apply(lambda df: pd.DataFrame({'maxB':df.maxB.iloc[0],
        'k':df.k.min()},index=[0])).reset_index(drop=True)
    res.insert(2,'minK',minK.k.iloc[minK.maxB.searchsorted(range(numBins),side='left')].values)

    maxK=pd.DataFrame({'minB':minB,'k':range(d)}).groupby(by='minB').apply(lambda df: pd.DataFrame({'minB':df.minB.iloc[0],
        'k':df.k.max()},index=[0])).reset_index(drop=True)
    res.insert(3,'maxK',maxK.k.iloc[maxK.minB.iloc[1:].searchsorted(range(numBins),side='right')].values)

    res=res.values
    
    print('ends', psutil.virtual_memory().percent)
    
    del minB, maxB
       
    z=np.abs(norm.ppf(mids/2))
    with ProcessPoolExecutor() as executor:    
        results=executor.map(vst, [(z[i*int(np.ceil(len(mids)/M)):min((i+1)*int(np.ceil(len(mids)/M)),len(mids))].tolist(),
            N,pairwise_cors.tolist()) for i in range(int(M))])
    print('vst', psutil.virtual_memory().percent)
        
    var=[]
    for result in results:
        var+=[result]
    
    var=np.concatenate(var,axis=0)
    var=var[np.argsort(-var[:,0]),1].reshape(-1,1) # sort according to lam / mids
    pd.DataFrame(np.concatenate([bins[1:].reshape(-1,1),var],axis=1),columns=['binEdges','var']).to_csv(
        sigName+'-'+str(N)+'-ebb-var.csv',index=False)

    rho = (var - N*mids*(1-mids)) / (N*(N-1)*mids*(1-mids))
    gamma = rho / (1-rho)

    res=np.concatenate([res,gamma,var],axis=1) # mids, binEdge,max,min, gamma, var
    
    print('gamma', psutil.virtual_memory().percent)
    with ProcessPoolExecutor() as executor: 
        results=executor.map(mp, [(res[i*int(np.ceil(len(res)/M)):min((i+1)*int(np.ceil(len(res)/M)),len(res)),:].tolist(),N) 
            for i in range(int(M))])
    
    ebb=[]
    for result in results:
        ebb+=result

    print('result', psutil.virtual_memory().percent)
    ebb=pd.concat(ebb,axis=0) # binEdge+k,ebb
    ebb=ebb.sort_values(by='sorter')
    bins=pd.Series(bins,name='bins')
    
    ebb.to_csv(sigName+'-'+str(N)+'-ebb-prob.csv',index=False)
    bins.to_csv(sigName+'-'+str(N)+'-ebb-binEdges.csv',index=False,header=True)
    pd.Series(pairwise_cors,name='pairwise_cors').to_csv(sigName+'-'+str(N)+'-ebb-pairwise_cors.csv',index=False)
    
    return(pairwise_cors)
        
def vst(dat):
    j=0
    z=np.array(dat[j]).reshape(-1,1);j+=1
    d=dat[j];j+=1
    pairwise_cors=np.array(dat[j]);j+=1
    
    return(np.concatenate([z,np.apply_along_axis(var_st,1, z,d=d,pairwise_cors=pairwise_cors)],axis=1))

def mp(dat):
    j=0;
    res=np.array(dat[j]);j+=1
    N=dat[j];j+=1
    cr=newC(N)

    ebb=[]    
    for row in res:
        j=0
        rLam=row[j];j+=1
        rBinEdge=row[j];j+=1
        rMin=int(row[j]);j+=1
        rMax=int(row[j]);j+=1
        rGamma=row[j];j+=1
        rVar=row[j];j+=1
        rLen=(rMax-rMin+1)

        if rGamma>=max(-rLam / (N-1),-(1-rLam) / (N-1)):   
            baseOne=np.append([0],np.cumsum(np.log(rLam+rGamma*np.array(range(0,rMax)))))
            baseTwo=np.cumsum(np.log(1-rLam+rGamma*np.array(range(N))))[-(rMax+1):][::-1]
            baseThree=np.sum(np.log(1+rGamma*np.array(range(N))))
            baseCr=cr[0:(int(rMax)+1)]

            Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
            ebb+=[pd.DataFrame({'sorter':[rBinEdge+x for x in range(rMin,rMax+1)],'ebb':
                (1-(np.sum(Pr[0:rMin])+np.cumsum(Pr[rMin:rMax+1])))},index=range(rLen))] # binEdge+k,ebb, var
             
        else:
            ebb+=[pd.DataFrame({'sorter':[rBinEdge+x for x in range(rMin,rMax+1)],'ebb':np.nan},index=range(rLen))] # binEdge+k,ebb

    return(ebb)