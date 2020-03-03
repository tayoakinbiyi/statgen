from opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb
import matplotlib.pyplot as plt
import os

def plotNullRef(parms):
    name=parms['name']
    numCores=parms['numCores']
    snpChr=parms['snpChr']
    
    Types=os.listdir('ref')
        
    fig,axs=plt.subplots(len(Types),1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    for TypeInd in range(len(Types)):  
        ref=np.loadtxt('ref/'+Types[TypeInd],delimiter='\t')
        stat=np.loadtxt('stats/'+Types[TypeInd]+'-'+str(2),delimiter='\t')
    
        axs[TypeInd].hist(ref,bins=40,label='ref',alpha=.25,density=True)
        axs[TypeInd].hist(stat,bins=40,label='sim',alpha=.25,density=True)
        axs[TypeInd].legend()
        axs[TypeInd].set_title(Types[TypeInd])
    
    fig.savefig('diagnostics/nullRef.png')
        
    return()
