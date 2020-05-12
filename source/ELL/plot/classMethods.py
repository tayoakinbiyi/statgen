import matplotlib.pyplot as plt
from zipfile import ZipFile
import pandas as pd
import numpy as np
from scipy.stats import beta
import pdb
from utility import *

def plot(self,pvals,file):
    qList=self.qList
        
    colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
    pvals=pd.Series(np.sort(pvals))    
    
    localLevels=pd.read_csv('../data/local_level_results.csv',header=0,index_col=None).values
    whichEta=np.argmin(np.abs(localLevels[:,0]-pvals.shape[0]))
    eta=localLevels[whichEta,1]
    
    M=len(pvals)
    nVec=np.arange(1,M+1)
    bounds=pd.DataFrame({'lower':beta.ppf(eta/2,nVec,nVec[::-1]),'upper':beta.ppf(1-eta/2,nVec,nVec[::-1])})
    
    pvals.index=np.arange(1,M+1)/(1+M)
    bounds.index=np.arange(1,M+1)/(1+M)

    pd.DataFrame(np.round(np.clip(np.concatenate([np.mean(pvals.values.reshape(-1,1)<qList,axis=0).reshape(-1,1),
                                                  np.concatenate([q+np.array([[-1,0,
        1]])*1.96*np.sqrt(q*(1-q)/m) for (qL,m) in ((qList,M),) for q in qL],axis=0)],axis=1),0,1),4),columns=['obs',
        'q-delta','q','q+delta'])[['q','q-delta','obs','q+delta']].to_csv(file+'.tsv',sep='\t')
    
    mMax=max(max(pvals.index.max(),pvals.max().max()),max(bounds.index.max(),bounds.max().max()))*1.1
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    pvals.plot(ax=axs,xlim=[0,mMax],ylim=[0,mMax],color=colors[0])
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    bounds.plot(ax=axs,xlim=[0,mMax],ylim=[0,mMax],color='black')
    fig.savefig(file+'-raw.png',bbox_inches='tight')
    
    log(file+' {} pvals out of bounds'.format(np.sum(((bounds['lower']>pvals)|(bounds['upper']<pvals))[0:int(.1*M)])))

    string='\n'.join(['\t'.join([str(elmt) for elmt in row]) for row in pd.concat([pvals,bounds],axis=1).values.tolist()])
    with ZipFile(file+'-raw.zip','w') as myZip:
        myZip.writestr(file+'-raw.tsv',string) 
      
    logPVals=pvals.copy()
    logBounds=bounds.copy()    
    logPVals.index=-np.log10(pvals.index)
    logPVals.iloc[:]=-np.log10(logPVals.iloc[:])
    logBounds.index=-np.log10(bounds.index)
    logBounds.iloc[:]=-np.log10(logBounds.iloc[:])
            
    mMax=max(max(logBounds.index.max(),logBounds.max().max()),max(logPVals.index.max(),logPVals.max().max()))*1.1
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    logPVals.plot(ax=axs,xlim=[0,mMax],ylim=[0,mMax],color=colors[0])
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    logBounds.plot(ax=axs,xlim=[0,mMax],ylim=[0,mMax],color='black')
    fig.savefig(file+'-log.png',bbox_inches='tight')
    
    plt.close('all')

    return()