import pandas as pd
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
import re
import numpy as np

from ail.opPython.DB import *

def plotOffDiag(source,parms):
    local=parms['local']
    name=parms['name']
    
    plotName=source[0]+' vs '+source[1]
            
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(20,forward=True)
    
    LCorr1=DBRead('LZCorr/'+source[0],parms)
    LCorr2=DBRead('LZCorr/'+source[1],parms)
    
    corr1=np.matmul(LCorr1,LCorr1.T)
    corr2=np.matmul(LCorr2,LCorr2.T)
    
    offDiag1=corr1[np.triu_indices(len(corr1),1)].flatten()
    offDiag2=corr2[np.triu_indices(len(corr2),1)].flatten()
    
    axs.scatter(offDiag1,offDiag2)
    axs.set_xlabel(source[0])
    axs.set_ylabel(source[1])
    axs.set_xlim([-1,1])
    axs.set_ylim([-1,1])
    axs.plot([-1,-1],[1,1],ls="--", c=".3")

    fig.savefig(local+name+'diagnostics/'+plotName+'.png')
    plt.close('all')        
    
    return()