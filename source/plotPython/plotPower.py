import matplotlib.pyplot as plt
from scipy.stats import beta
from zipfile import ZipFile

import pandas as pd
import numpy as np

def plotPower(pvals,file,qList=[.01,.001]):
    colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
        
    type1=pd.DataFrame([np.mean(pvals<q,axis=0) for q in qList],columns=pvals.columns,index=qList)
    type1.index.name='alpha'
    type1=type1.reset_index()
    type1.to_csv('diagnostics/type1-'+prefix+'.tsv',sep='\t')
    
    pvals=pd.DataFrame(np.sort(pvals,axis=0), columns=pvals.columns)
    cols=pvals.columns.values
    cols[crossMetrics>0]+='-c'
    pvals.columns=cols
    
    localLevels=pd.read_csv('../data/local_level_results.csv',header=0,index_col=None).values
    whichEta=np.argmin(np.abs(localLevels[:,0]-pvals.shape[0]))
    eta=localLevels[whichEta,1]
    
    M=len(pvals)
    nVec=np.arange(1,M+1)
    bounds=pd.DataFrame({'lower':beta.ppf(eta/2,nVec,nVec[::-1]),'upper':beta.ppf(1-eta/2,nVec,nVec[::-1])})
    
    pvals.index=np.arange(1,M+1)/(1+M)
    bounds.index=np.arange(1,M+1)/(1+M)
    crossMetrics=((pvals.values.reshape(-1,pvals.shape[1])<=bounds['lower'].values.reshape(-1,1))|(pvals.values.reshape(-1,
        pvals.shape[1])>=bounds['upper'].values.reshape(-1,1))).sum(axis=0)
    
    mMax=max(pvals.index.max(),pvals.max().max())*1.1
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    pvals.plot(ax=axs,legend=True,xlim=[0,mMax],ylim=[0,mMax],color=colors[0:pvals.shape[1]])
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    bounds.plot(ax=axs,legend=False,xlim=[0,mMax],ylim=[0,mMax],color='black')
    axs.set_title(file)
    fig.savefig('diagnostics/'+file+'.png',bbox_inches='tight')
    
    pd.concat([pvals,bounds],axis=1).to_csv(file+'.tsv',sep='\t')    
    with ZipFile('diagnostics/'+file+'.zip','w') as myZip:
        myZip.write(file+'.tsv') 
        
    logPVals=pvals.copy()
    logBounds=bounds.copy()    
    logPVals.index=-np.log10(pvals.index)
    logPVals.iloc[:]=-np.log10(pvals.iloc[:])
    logBounds.index=-np.log10(bounds.index)
    logBounds.iloc[:]=-np.log10(bounds.iloc[:])
        
    mMax=max(pvals.index.max(),pvals.max().max())*1.1
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)    
    pvals.plot(ax=axs,legend=True,xlim=[0,mMax],ylim=[0,mMax],color=colors[0:pvals.shape[1]])
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    bounds.plot(ax=axs,legend=False,xlim=[0,mMax],ylim=[0,mMax],color='black')
    axs.set_title(file+'-log')
    fig.savefig('diagnostics/'+file+'-log.png',bbox_inches='tight')
    
    plt.close('all')

    return(crossMetrics>0,type1)

