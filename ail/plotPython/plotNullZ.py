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
    numQSnps=parms['numQSnps']

    mean=np.array([])
    mean2=np.array([])
    mean4=np.array([])
    
    for trait in traitChr:
        fig,axs=plt.subplots(4,1,dpi=150)
        fig.set_figwidth(20,forward=True)
        fig.set_figheight(80,forward=True)
        fig.tight_layout()

        mean=DBRead(name+'corr/mean-'+trait,parms,toPickle=True).flatten()
        mean2=DBRead(name+'corr/mean2-'+trait,parms,toPickle=True).flatten()
        mean4=DBRead(name+'corr/mean4-'+trait,parms,toPickle=True).flatten()
        
        z=[]
        for snp in snpChr:
            print('for plotNullZ reading z scores '+snp+' '+trait,flush=True)
            z+=[DBRead(name+'score/p-'+snp+'-'+trait,parms,toPickle=True).astype('float16').flatten()]
            
        val=np.concatenate(z).astype('float16')
        if len(val)>numQSnps:
            qMesh=np.linspace(0,1,numQSnps+2)[1:-1]
            obsQ=np.sort(val)[np.round(qMesh*len(val)).astype(int)-1]
        else:
            qMesh=np.linspace(0,1,len(val)+2)[1:-1]
            obsQ=np.sort(val)

        expQ=norm.ppf(qMesh)             
        
        minQ=np.min([np.mininimum(expQ),np.minimum(obsQ)])
        maxQ=np.max([np.maximum(expQ),np.maximum(obsQ)])
        
        axs[0].scatter(expQ,obsQ,s=.01)
        axs[0].set_xlim([minQ,maxQ])
        axs[0].set_ylim([minQ,maxQ])
        axs[0].tick_params(axis='both', which='major', labelsize=20)
        axs[0].set_xlabel('expected')
        axs[0].set_ylabel('actual')
        axs[1].hist(mean,bins=100)
        axs[1].tick_params(axis='both', which='major', labelsize=20)
        axs[2].hist(mean2,bins=100)
        axs[2].tick_params(axis='both', which='major', labelsize=20)
        axs[3].hist(mean4,bins=100)
        axs[3].tick_params(axis='both', which='major', labelsize=20)

        fig.savefig(local+name+'plots/plotNullZ-'+trait+'.png',bbox_inches='tight')

        plt.close('all')
        DBUpload(name+'plots/plotNullZ-'+trait+'.png',parms,toPickle=False)
    
    return()
    