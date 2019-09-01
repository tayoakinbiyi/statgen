import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
import re

from ail.dataPrepPython.genCorr import *
   
def plotCorr(parms):
    plotsDir=parms['plotsDir']
    response=parms['response']
    scratchDir=parms['scratchDir']
    traitChr=parms['traitChr']
    
    corr=genCorr('',parms)    
    
    print('make response corr',flush=True)
    if not os.path.exists(scratchDir+'corr-response.csv'):
        expr=np.loadtxt(scratchDir+'trait.csv',delimiter=',')
        obsCorr=np.corrcoef(expr,rowvar=False)
        np.savetxt(scratchDir+'corr-response.csv',obsCorr,delimiter=',')
    else:
        obsCorr=np.loadtxt(scratchDir+'corr-response.csv',delimiter=',')
        
    fig,axs=plt.subplots(1,3)
    fig.set_figwidth(21,forward=True)
    fig.set_figheight(7,forward=True)
    
    print('from z scores hist', flush=True)
    off_diag=corr[np.triu_indices(len(corr),1)].flatten()  
    axs[0].hist(off_diag,bins=np.linspace(-1,1,1000),log=True)
    axs[0].set_title('from z scores')
    
    print('from response hist',flush=True)
    obs_off_diag=obsCorr[np.triu_indices(len(obsCorr),1)].flatten()  
    axs[1].hist(obs_off_diag,bins=np.linspace(-1,1,1000),log=True)
    axs[1].set_title('from response parms')
    
    print('z scores vs response',flush=True)
    axs[2].scatter(obs_off_diag,off_diag)
    axs[2].set_title('from z score vs from response file')
    axs[2].set_xlabel('from response file')
    axs[2].set_ylabel('from z score')

    fig.savefig(plotsDir+'full_corr.png',bbox_inches='tight')
    plt.close('all')        