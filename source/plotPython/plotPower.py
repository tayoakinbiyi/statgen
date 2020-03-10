from opPython.DB import *
import matplotlib.pyplot as plt
from scipy.stats import beta
import statsmodels.api as sm

import pandas as pd
import numpy as np

def plotPower(pvals,parms,title,columns,log=True,myZip=None):
    local=parms['local']
    colors=parms['colors']
    
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    
    pvals=pd.DataFrame(np.sort(pvals,axis=0), columns=columns)
    
    M=len(pvals)
    
    localLevels=pd.read_csv(local+'data/local_level_results.csv',header=0,index_col=None).values
    eta=localLevels[np.argmin(np.abs(localLevels[:,0]-len(pvals))),1]
    nVec=np.arange(1,M+1)
    bounds=pd.DataFrame({'lower':beta.ppf(eta/2,nVec,nVec[::-1]),'upper':beta.ppf(1-eta/2,nVec,nVec[::-1])})
    
    pvals.index=np.arange(1,M+1)/(1+M)
    bounds.index=np.arange(1,M+1)/(1+M)
    crossMetrics=((pvals.values.reshape(-1,pvals.shape[1])<=bounds['lower'].values.reshape(-1,1))|(pvals.values.reshape(-1,
        pvals.shape[1])>=bounds['upper'].values.reshape(-1,1))).sum(axis=0)
    
    if log:
        pvals.index=-np.log10(pvals.index)
        pvals.iloc[:]=-np.log10(pvals.iloc[:])
        bounds.index=-np.log10(bounds.index)
        bounds.iloc[:]=-np.log10(bounds.iloc[:])
        
    cols=pvals.columns.values
    cols[crossMetrics>0]+='-c'
    pvals.columns=cols

    mMax=max(pvals.index.max(),pvals.max().max())*1.1
    pvals.plot(ax=axs,legend=True,xlim=[0,mMax],ylim=[0,mMax],color=colors[0:pvals.shape[1]])
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    bounds.plot(ax=axs,legend=False,xlim=[0,mMax],ylim=[0,mMax],color='black')
    axs.set_title(title)
    
    pvals=pd.concat([pvals,bounds],axis=1)
    pvals=np.concatenate([pvals.columns.values.reshape([1,-1]),pvals.values],axis=0)
    
    if not myZip is None:
        myZip.writestr(title,'\n'.join(map(lambda x:','.join(map(str,x)),pvals.tolist()))) 
        
    fig.savefig('diagnostics/'+title+'.png',bbox_inches='tight')

    return(crossMetrics>0)

