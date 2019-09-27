import pandas as pd
import numpy as np
import pdb
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
from ail.opPython.DB import *

def plotZ(parms):
    plt.rcParams.update({'font.size': 20})
    
    name=parms['name']
    local=parms['local']
    
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    smallCpu=parms['smallCpu']
    numQSnps=parms['numQSnps']

    mean=np.array([])
    mean2=np.array([])
    mean4=np.array([])
    
    for trait in traitChr:
        fig,axs=plt.subplots(4,1,dpi=150)
        fig.set_figwidth(20,forward=True)
        fig.set_figheight(80,forward=True)

        mean=DBRead(name+'corr/mean-'+trait,parms,toPickle=True).flatten()
        mean2=DBRead(name+'corr/mean2-'+trait,parms,toPickle=True).flatten()
        mean4=DBRead(name+'corr/mean4-'+trait,parms,toPickle=True).flatten()
        
        z=[]
        for snp in snpChr:
            print('for plotZ reading z scores '+snp+' '+trait,flush=True)
            z+=[DBRead(name+'score/p-'+snp+'-'+trait,parms,toPickle=True).astype('float16').flatten()]
            
        val=np.concatenate(z).astype('float16')
        if len(val)>numQSnps:
            qMesh=np.linspace(0,1,numQSnps+2)[1:-1]
            obsQ=np.sort(val)[np.round(qMesh*len(val)).astype(int)-1]
        else:
            qMesh=np.linspace(0,1,len(val)+2)[1:-1]
            obsQ=np.sort(val)

        expQ=norm.ppf(qMesh)             
        
        allQ=expQ.tolist()+obsQ.tolist()
        minQ=np.min(allQ)
        maxQ=np.max(allQ)
        
        del allQ
        
        axs[0].scatter(expQ,obsQ,s=.3,color='red')
        axs[0].set_xlim([minQ,maxQ])
        axs[0].set_ylim([minQ,maxQ])
        axs[0].set_xlabel('expected')
        axs[0].set_ylabel('actual')
        axs[0].set_title('qq plot vs N(0,1)')
        axs[0].plot(axs[0].get_xlim(), axs[0].get_ylim(), ls="--", c='k')

        axs[1].hist(mean,bins=100)
        axs[1].set_title('first moment plot')

        axs[2].hist(mean2,bins=100)
        axs[2].set_title('second moment plot')

        axs[3].hist(mean4,bins=100)
        axs[3].set_title('fourth moment plot')

        fig.savefig(local+name+'plots/H0-Z-Scores-trait:'+trait+'.png')

        plt.close('all')
        DBUpload(name+'plots/H0-Z-Scores-trait:'+trait+'.png',parms,toPickle=False)
    
    return()
    