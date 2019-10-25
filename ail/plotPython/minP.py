import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from ail.opPython.DB import *

def minP(parms):
    local=parms['local']
    name=parms['name']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    numCores=parms['numCores']
    
    DBCreateFolder(name,'minP',parms)

    allMin=[]
    cisMin=[]
    futures=[]
    with ProcessPoolExecutor(numCores) as executor:
        for trait in traitChr:
            futures+=[executor.submit(minPHelp,trait,parms)]
            
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            tAllMin,tCisMin=f.result()
            allMin+=[tAllMin]
            cisMin+=[tCisMin]
            
    allMin=np.concatenate(allMin,axis=1).flatten()
    cisMin=np.concatenate(cisMin,axis=1).flatten()

    fig,axs=plt.subplots(1,1,dpi=50)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(20,forward=True)

    axs.set_title('minP')
    axs.hist(-np.log10(cisMin),bins=100,label='cis')
    axs.hist(-np.log10(allMin),bins=100,label='all')
    
    axs.legend()
            
    fig.savefig(local+name+'minP/minP.png',bbox_inches='tight')
    plt.close('all')

    DBUpload(name+'minP/minP.png',parms,False)
    print('wrote minP',flush=True)
    
    return()

def minPHelp(trait,parms):
    snpChr=parms['snpChr']
    name=parms['name']
    wald=parms['wald']
    
    tmp=[]
    for snp in snpChr:
        print('loading pvals from snp '+snp+' trait '+trait)
        if wald:
            df=2*norm.sf(np.abs(DBRead(name+'score/p-'+snp+'-'+trait,parms,True)))
        else:
            df=DBRead(name+'score/p-'+snp+'-'+trait,parms,True)
            
        df=np.min(df,axis=0).reshape(1,-1)
        tmp+=[df]
        if snp==trait:
            cisMin=df.reshape(1,-1)
    allMin=np.min(np.concatenate(tmp,axis=0),axis=0).reshape(1,-1)
    
    return(allMin,cisMin)
