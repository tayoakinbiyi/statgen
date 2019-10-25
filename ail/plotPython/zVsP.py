import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED

def qqPlots(parms):
    plotsDir=parms['plotsDir']
    scratchDir=parms['scratchDir']    
    traitChr=parms['traitChr']
    smallCpu=parms['smallCpu']
    
    fig,axs=plt.subplots(len(traitChr),1,dpi=50)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(190,forward=True)

    for i in range(len(traitChr)):
        trait=traitChr[i]

        if not (trait in snpChr):
            cisTrait='chr1'
        else:
            cisTrait=trait

        transP=[]
        transZ=[]
        for snp in snpChr:
            print('loading pvals from snp '+snp+' trait '+trait)
            dfP=np.loadtxt(scratchDir+'p-'+snp+'-'+trait+'.csv',delimiter=',')[:,0:50]
            dfZ=2*np.sf(np.abs(np.loadtxt(scratchDir+'z-'+snp+'-'+trait+'.csv',delimiter=',')[:,0:50]))
            if snp==cisTrait:
                cisP=dfP
                cisZ=dfZ
            else:
                transP+=[dfP]
                transZ+=[dfZ]

        transZ=-np.log10(np.concatenate(np.split(np.concatenate(transZ,axis=0),1)))
        transP=-np.log10(np.concatenate(np.split(np.concatenate(transP,axis=0),1)))
        
        cisP=-np.log10(np.concatenate(np.split(cisP,1)))
        cisZ=-np.log10(np.concatenate(np.split(cisZ,1)))
        
        axs[i].set_xlim([np.min([transZ.min(),cisZ.min()]),np.max([transZ.max(),cisZ.max()])])
        axs[i].set_xlim([np.min([transP.min(),cisP.min()]),np.max([transP.max(),cisP.max()])])

        axs[i].scatter(transZ,transP,label='trans')
        axs[i].scatter(cisZ,cisP,label='cis')
        axs[i].set_ylabel(trait+' P from LRT')
        axs[i].set_xlabel('P from Z score')
        axs[i].plot(axs[i].get_xlim(), axs[i].get_ylim(), ls="--", c=".3")    
        axs[i].legen()
        
    fig.savefig(plotsDir+'zVsP.png',bbox_inches='tight')
    plt.close('all')
    
