import numpy as np
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm, beta
from qq_var import *
import pdb

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
    return(pd.DataFrame({'mids':df.mids.iloc[0],'min':df.k.min(),'max':df.k.max()},index=[0]))

def makeProb(L,t_mu,t_eps,size,pairwise_cors,N,name):
    #if  os.path.isfile(name+'-ebb.csv'):
    #    return()
    z=np.matmul(L.T,np.random.normal(0,1,size=(N,500))).T

    d=int(N/2)
    cr=newC(N)

    M=multiprocessing.cpu_count()
    
    res=[]
    
    for mu in t_mu:
        for eps in np.linspace(2,N*(.008 if N>2000 else .01 if N>1000 else .017),10).astype(int):
            t_z=z.copy()
            if mu*eps>0:
                t_z[:,range(eps)]+=mu
            lam=np.sort((2*norm.sf(np.abs(t_z))))[:,0:d].reshape(-1,1)
            res+=[np.concatenate([lam.reshape(-1,1),[[x] for x in range(d)]*len(t_z)],axis=1)] # lam,k
            
    res=np.concatenate(res,axis=0)
    bins=np.histogram_bin_edges(res[:,0],bins=int(N**1.5))
    bins=np.append(np.append(np.linspace(0,bins[2],100),bins[3:-2]),np.linspace(bins[-2],1,100))
    mids=(bins[1:]+bins[:-1]).reshape(-1,1)/2
   
    res=np.concatenate([res[:,1].reshape(-1,1),(np.digitize(res[:,0],bins)-1).reshape(-1,1)],axis=1) #[k,bins] 
    res=pd.DataFrame(res,columns=['k','bins']).groupby('k').apply(binsMinMax,B=len(mids),N=N).reset_index(drop=True)
    res.insert(2,'mids',mids[res.bins.astype(int)])
    res.drop(columns='bins',inplace=True)    
    res=res.groupby('mids',sort=False).apply(kMinMax).reset_index(drop=True)
    res=res.sort_values(by='mids').values # mids, min, max
    res=np.concatenate([res,bins[1:].reshape(-1,1)],axis=1) # mids, min, max, binEdge
    
    t_z=np.abs(norm.ppf(mids/2))
    with ProcessPoolExecutor() as executor:    
        results=executor.map(vst, [(t_z[i*int(np.ceil(len(mids)/M)):min((i+1)*int(np.ceil(len(mids)/M)),len(mids))],
            N,pairwise_cors) for i in range(int(M))])
        
    var=[]
    for result in results:
        var+=[result]

    var=np.concatenate(var,axis=0)
    var=var[np.argsort(-var[:,0]),1].reshape(-1,1) # sort according to lam / mids
    rho = (var - N*mids*(1-mids)) / (N*(N-1)*mids*(1-mids))
    gamma = rho / (1-rho)

    res=np.concatenate([res,gamma,var],axis=1) # mids, min,max,binEdge, gamma, var
    
    #pdb.set_trace()
    #i=0
    #mp((res,N,cr))
    
    with ProcessPoolExecutor() as executor: 
        results=executor.map(mp, [(res[i*int(np.ceil(len(res)/M)):min((i+1)*int(np.ceil(len(res)/M)),len(res)),:],N,cr) 
            for i in range(int(M))])
    
    ebb=[]
    for result in results:
        ebb+=result

    ebb=pd.concat(ebb,axis=0) # binEdge+k,ebb
    ebb=ebb.sort_values(by='sorter')
    bins=pd.Series(bins,name='bins')
    
    ebb.to_csv(name+'-ebb.csv',index=False)
    bins.to_csv(name+'-ebb-binEdges.csv',index=False,header=True)
    
    return()
        
def vst(dat):
    j=0
    z=dat[j].reshape(-1,1);j+=1
    d=dat[j];j+=1
    pairwise_cors=dat[j];j+=1
    
    return(np.concatenate([z,np.apply_along_axis(var_st,1, z,d=d,pairwise_cors=pairwise_cors)],axis=1))

def mp(dat):
    j=0;
    res=dat[j];j+=1
    N=dat[j];j+=1
    cr=dat[j];j+=1
    # res:  mids, min,max,gamma,binEdge
    #pdb.set_trace()
    ebb=[]    
    for row in res:
        j=0
        rLam=row[j];j+=1
        rMin=int(row[j]);j+=1
        rMax=int(row[j]);j+=1
        rBinEdge=row[j];j+=1
        rGamma=row[j];j+=1
        rVar=row[j];j+=1
        rLen=(rMax-rMin+1)

        if rGamma>=max(-rLam / (N-1),-(1-rLam) / (N-1)):   
            baseOne=np.append([0],np.cumsum(np.log(rLam+rGamma*np.array(range(0,rMax)))))
            baseTwo=np.cumsum(np.log(1-rLam+rGamma*np.array(range(N))))[-(rMax+1):][::-1]
            baseThree=np.sum(np.log(1+rGamma*np.array(range(N))))
            baseCr=cr[0:(int(rMax)+1)]

            Pr=np.exp(baseCr+baseOne+baseTwo-baseThree)
            ebb+=[pd.DataFrame({'sorter':[rBinEdge+x for x in range(rMin,rMax+1)],'k':range(rMin,rMax+1),'ebb':
                (1-(np.sum(Pr[0:rMin])+np.cumsum(Pr[rMin:rMax+1]))),'var':rVar},index=range(rLen))] # binEdge+k,ebb, var
             
        else:
            ebb+=[pd.DataFrame({'sorter':[rBinEdge+x for x in range(rMin,rMax+1)],'k':range(rMin,rMax+1),'ebb':np.nan,
                  'var':rVar},index=range(rLen))] # binEdge+k,np.nan, var

    return(ebb)