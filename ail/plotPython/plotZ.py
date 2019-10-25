import pandas as pd
import numpy as np
import pdb
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
from ail.opPython.DB import *
import scipy.stats
import statsmodels.api as sm
from ail.dataPrepPython.genSnpMeans import *

def plotZ(parms,nameParm=''):
    name=parms['name']
    local=parms['local']
    
    DBCreateFolder('plotZ',parms)
    plt.rcParams.update({'font.size': 20})
    
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']

    df=[]
    for trait in traitChr:
        snpDF=[]
        for snp in snpChr:
            snpDF+=[DBRead(name+'score/'+trait,parms)]
        df+=[np.concatenate(snpDF,axis=1)]
        
    df=np.concatenate(df,axis=0)

    mean=df.mean(axis=0)
    mean2=(df**2).mean(axis=0)
    mean3=(df**3).mean(axis=0)
    mean4=(df**4).mean(axis=0)
    
    fig,axs=plt.subplots(5,1,dpi=50)
    fig.set_figwidth(100,forward=True)
    fig.set_figheight(20,forward=True)

    j=0        
    sm.qqplot(-np.log10(p),scipy.stats.expon,ax=axs[j],loc=0,scale=np.log10(np.exp(1)))
    axs[j].plot(axs[j].get_xlim(), axs[j].get_ylim(), ls="--", c='k')

    j+=1
    axs[j].hist(mean,bins=100)
    axs[j].set_title('first moment plot')

    j+=1
    axs[j].hist(mean2,bins=100)
    axs[j].set_title('second moment plot')

    j+=1
    axs[j].hist(mean3,bins=100)
    axs[j].set_title('third moment plot')

    j+=1
    axs[j].hist(mean4,bins=100)
    axs[j].set_title('fourth moment plot')

    fig.savefig('plotZ/all.png')
    plt.close('all')
    
    print('finished plot Z',flush=True)
    return()
    