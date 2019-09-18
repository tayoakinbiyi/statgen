import pandas as pd
import numpy as np
import pdb
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
from ail.opPython.DB import *

def plotNullZ(parms):
    name=parms['name']
    local=parms['local']
    
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    smallCpu=parms['smallCpu']

    mean=np.array([])
    mean2=np.array([])
    mean4=np.array([])
    
    fig,axs=plt.subplots(len(traitChr),4,dpi=50)
    fig.set_figwidth(80,forward=True)
    fig.set_figheight(20*len(traitChr),forward=True)
    fig.tight_layout()

    for trait in range(len(traitChr)):
        mean=DBRead(name+'corr/mean-'+traitChr[trait],parms,toPickle=True).flatten()
        mean2=DBRead(name+'corr/mean2-'+traitChr[trait],parms,toPickle=True).flatten()
        mean4=DBRead(name+'corr/mean4-'+traitChr[trait],parms,toPickle=True).flatten()
        
        z=[]
        for snp in snpChr:
            print('for plotNullZ reading z scores '+snp+' '+traitChr[trait],flush=True)
            z+=[DBRead(name+'score/p-'+snp+'-'+traitChr[trait],parms,toPickle=True).flatten()]
            
        z=np.sort(np.concatenate(z))
        
        exp=norm.ppf(np.arange(1,len(z)+1)/(len(z)+1))   
        
        axs[trait,0].scatter(exp,z,s=.01)
        axs[trait,0].set_title('qq vs N(0,1)')
        axs[trait,1].hist(mean,bins=100)
        axs[trait,2].hist(mean2,bins=100)
        axs[trait,3].hist(mean4,bins=100)

    fig.savefig(local+name+'plots/plotNullZ.png',bbox_inches='tight')

    plt.close('all')
    DBUpload(name+'plots/plotNullZ.png',parms,toPickle=False)
    
    return()
    