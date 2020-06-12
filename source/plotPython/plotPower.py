import matplotlib.pyplot as plt
from zipfile import ZipFile
import pandas as pd
import numpy as np
from scipy.stats import beta, binom
import pdb
from utility import *

def plot(pvals,file,cols,qList=np.array([0.01,0.001])):
    path='diagnostics/'+file

    with ZipFile(path+'-data.zip','w') as myZip:
        myZip.writestr(file+'-data.tsv',arrString(pvals)) 
    
    localLevels=pd.read_csv('../data/local_level_results.csv',header=0,index_col=None).values
    whichEta=np.argmin(np.abs(localLevels[:,0]-pvals.shape[0]))
    eta=localLevels[whichEta,1]
    
    pvals=pd.DataFrame(np.sort(pvals,axis=0),columns=cols)
    
    M=len(pvals)
    nVec=np.arange(1,M+1)
    bounds=pd.DataFrame({'lower':beta.ppf(eta/2,nVec,nVec[::-1]),'upper':beta.ppf(1-eta/2,nVec,nVec[::-1])})
    pvals.index=np.arange(1,M+1)/(1+M)
    bounds.index=np.arange(1,M+1)/(1+M)
    
    table=np.concatenate([
            np.array([['q','0.025','0.975']+cols]),
            np.concatenate([
                qList.reshape(-1,1),
                np.concatenate([binom.ppf([0.025,0.975],m,q).reshape(1,2)/m for (qL,m) in ((qList,M),) for q in  qL],axis=0)
            ]+[
                np.round(np.mean(x.reshape(-1,1)<qList.reshape(1,-1),axis=0).reshape(-1,1),4) for x in pvals.values.T]
            ,axis=1),
        ],axis=0)
    np.savetxt(path+'-table.tsv',table,delimiter='\t',fmt='%s')
    
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    pvals.plot(ax=axs,xlim=[0,1],ylim=[0,1],legend=True)
    axs.plot([0,1], [0,1], ls="--", c=".3")   
    bounds.plot(ax=axs,xlim=[0,1],ylim=[0,1],color='black',legend=False)
    fig.savefig(path+'-raw.png',bbox_inches='tight')
    
    lift=pd.DataFrame(np.concatenate([
        (1+np.searchsorted(x,np.arange(1,len(x)+1)/(len(x)+1))).reshape(-1,1)/(len(x)+1) for x in pvals.values.T
    ],axis=1),index=pvals.index,columns=cols)
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    lift.plot(ax=axs,xlim=[0,1],ylim=[0,1.1],legend=True)
    axs.plot(lift.index,lift.index,ls='--',c='.3')
    fig.savefig(path+'-lift.png',bbox_inches='tight')
    
    liftLog=pd.DataFrame(np.concatenate([
        (1+np.searchsorted(x,np.arange(1,len(x)+1)/(len(x)+1))).reshape(-1,1)/(len(x)+1) for x in pvals.values.T
    ],axis=1),index=-np.log10(pvals.index),columns=cols)
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    liftLog.plot(ax=axs,ylim=[0,1.1],legend=True)
    axs.plot(liftLog.index,-np.log10(liftLog.index),ls='--',c='.3')
    fig.savefig(path+'-lift-log.png',bbox_inches='tight')

    logPVals=pvals.copy()
    logBounds=bounds.copy()    
    logPVals.index=-np.log10(pvals.index)
    logPVals.iloc[:]=-np.log10(logPVals.iloc[:])
    logBounds.index=-np.log10(bounds.index)
    logBounds.iloc[:]=-np.log10(logBounds.iloc[:])
            
    mMax=max(logBounds.index.max(),logPVals.index.max())*1.1
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    logPVals.plot(ax=axs,xlim=[0,mMax],legend=True)
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    logBounds.plot(ax=axs,xlim=[0,mMax],color='black',legend=False)
    fig.savefig(path+'-log.png',bbox_inches='tight')
        
    plt.close('all')

    return(table[1,3:].astype(float))