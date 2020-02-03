from opPython.DB import *
import matplotlib.pyplot as plt
from scipy.stats import beta
import statsmodels.api as sm

import pandas as pd
import numpy as np

def plotPower(pvals,parms):
    local=parms['local']
    colors=parms['colors']
    
    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    
    M=len(pvals)
    
    localLevels=pd.read_csv(local+'data/local_level_results.csv',header=0,index_col=None).values
    eta=localLevels[np.argmin(np.abs(localLevels[:,0]-len(pvals))),1]
    nVec=np.arange(1,M+1)
    bounds=pd.DataFrame({'lower':-np.log10(beta.ppf(eta/2,nVec,nVec[::-1])),'upper':-np.log10(beta.ppf(1-eta/2,nVec,nVec[::-1]))})
    
    pvals.index=-np.log10(np.arange(1,M+1)/(1+M))
    bounds.index=-np.log10(np.arange(1,M+1)/(1+M))

    mMax=max(pvals.index.max(),pvals.max().max())*1.1
    pvals.plot(ax=axs,legend=True,xlim=[0,mMax],ylim=[0,mMax],color=colors[0:pvals.shape[1]])
    axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
    bounds.plot(ax=axs,legend=False,xlim=[0,mMax],ylim=[0,mMax],color='black')
            
    fig.savefig('diagnostics/power.png',bbox_inches='tight')

    return()

