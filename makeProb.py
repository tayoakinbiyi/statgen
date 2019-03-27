import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.stats import norm, beta
from qq_var import *
import pdb
import psutil
import matplotlib.pylab as plt
import time
from ggof import *
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
    binPower=500
    
    N=parms['N']
    sigName=parms['sigName']
    
    d=int(N/2)

    z=np.matmul(L.T,np.random.normal(0,1,size=(N,200))).T
    pairwise_cors=np.array([0]*int((N-1)*N/2)) #np.corrcoef(z,rowvar=False)[np.triu_indices(N,1)].flatten()   

    if os.path.isfile('ebb/'+sigName+'/ebb.csv'):
        return()
    else:
        try:
            os.mkdir('ebb/'+sigName)
        except:
            print('directory already exists')
    
    M=multiprocessing.cpu_count()
    
    numBins=int(N*binPower)
    t0=time.time()
    print('start',round((time.time()-t0),2))
    
    bins=np.linspace(0,.5,numBins+1)
    bins=np.concatenate([np.linspace(0,bins[1],500),bins[2:-1],np.linspace(bins[-1],1,500)]).flatten()
    
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

    minMaxK=minMaxK.values
    
    print('ends', psutil.virtual_memory().percent,round((time.time()-t0),2))
          
    z=np.abs(norm.ppf(bins/2))
    rhoBar=getRhoBar(pairwise_cors)
    with ProcessPoolExecutor() as executor:    
        results=executor.map(vst, [(z[i*int(np.ceil(numBins/M)):min((i+1)*int(np.ceil(numBins/M)),numBins)].tolist(),N,rhoBar) 
            for i in range(int(np.ceil(numBins/np.ceil(numBins/M))))])
        
    print('vst', psutil.virtual_memory().percent,round((time.time()-t0),2))
        
    res=[]
    for result in results:
        res+=[result]
    
    var=np.concatenate(res)
    pd.DataFrame(np.concatenate([bins.reshape(-1,1),var.reshape(-1,1)],axis=1),columns=['binEdges','var']).to_csv(
        'ebb/'+sigName+'/var.csv',index=False)

    rho = (var - N*bins*(1-bins)) / (N*(N-1)*bins*(1-bins))
    gamma = rho / (1-rho)

    minMaxK=np.concatenate([minMaxK,gamma.reshape(-1,1),var.reshape(-1,1)],axis=1) # binEdge,min,max, gamma, var
    print('gamma', psutil.virtual_memory().percent,round((time.time()-t0),2))

    with ProcessPoolExecutor() as executor: 
        results=executor.map(mp, [(minMaxK[i*int(np.ceil(numBins/M)):min((i+1)*int(np.ceil(numBins/M)),numBins)].tolist(),N) 
            for i in range(int(np.ceil(numBins/np.ceil(numBins/M))))])
    
    print('mp', psutil.virtual_memory().percent,round((time.time()-t0),2))

    res=[]
    for result in results:
        res+=result

    ebb=np.array(res) # binEdge,k,ebb
    ebb=ebb[np.argsort(ebb[:,0]+ebb[:,1])][:,[0,2]]
    
    print('ebb', psutil.virtual_memory().percent,round((time.time()-t0),2))

    end=np.cumsum(maxB-minB+1).reshape(-1,1)
    start=np.append([0],end[:-1]).reshape(-1,1)
    minMaxB=np.concatenate([start,end-1],axis=1)
    '''arr,cr=ggStats(N)
    pdb.set_trace()    
    b=0;lam=np.append(ebb[minMaxB[:,0]+b,0],[.9]*d);print(ggof(np.abs(norm.ppf(lam/2)),lam,pairwise_cors,arr,cr)['ggnull'],'\n',ebb[minMaxB[:,0]+b,1])'''
    
    np.savetxt('ebb/'+sigName+'/minMaxB.csv',minMaxB,delimiter=',')
    pd.Series(rhoBar,name='rhoBar').to_csv('ebb/'+sigName+'/rhoBar.csv',index=False,header=True)
    np.savetxt('ebb/'+sigName+'/ebb.csv',ebb,delimiter=',')  
    
    return(pairwise_cors)
           
def saveEBB(dat):
    j=0
    name=dat[j];j+=1
    df=pd.DataFrame(dat[j],columns=['binEdge','ebb']);j+=1
    print(name)
    df.to_csv(name,index=False)
    
def vst(dat):
    j=0
    z=np.array(dat[j]).flatten();j+=1
    d=dat[j];j+=1
    rhoBar=dat[j];j+=1
    
    return(getVarNoMu(z,d,rhoBar))
    
def mp(dat):
    j=0;
    res=np.array(dat[j]);j+=1
    N=dat[j];j+=1
    
    cr=newC(N)

    ebb=[]    
    for row in res:
        j=0
        rLam=row[j];j+=1
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
            ans=1-(np.sum(Pr[0:rMin])+np.cumsum(Pr[rMin:rMax+1]))
            ebb+=[[rLam,rMin+k,ans[k]] for k in range(rMax-rMin+1)] 
        else:
            ebb+=[[rLam,rMin+k,np.nan] for k in range(rMax-rMin+1)] # binEdge+k,ebb
    
    return(ebb)