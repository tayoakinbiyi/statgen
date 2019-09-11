import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
import re

from ail.dataPrepPython.genCorr import *
   
def plotCorr(source,plotName,parms):
    local=parms['local']
    name=parms['name']
    
    keys=list(source.keys())
    offDiag={}
            
    fig,axs=plt.subplots(3,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(21,forward=True)
    
    for key in [0,1]:
        LCorr=DBRead(name+source[keys[key]],parms,toPickle=True)
        corr=np.matmul(LCorr,LCorr.T)
        offDiag[keys[key]]=corr[np.triu_indices(len(corr),1)].flatten()
        axs[key].hist(off_diag[keys[key]],bins=np.linspace(-1,1,1000),log=True)
        axs[key].set_title(keys[key])
    
    axs[2].scatter(offDiag[keys[0]],offDiag[keys[1]])
    axs[2].set_xlabel(keys[0])
    axs[2].set_ylabel(keys[1])

    fig.savefig(local+name+'plots/'+plotName+'.png',bbox_inches='tight')
    plt.close('all')        
    
    DBUpload(name+'plots/'+plotName+'.png',parms,toPickle=False)

    return(offDiag[keys[0]],offDiag[keys[1]])