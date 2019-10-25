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

def qqPlots(parms):
    local=parms['local']
    name=parms['name']
    traitChr=parms['traitChr']
    smallCpu=parms['smallCpu']
    
    DBCreateFolder(name,'qqPlots',parms)

    futures=[]
    #qqPlotsHelp('chr2',parms)
    with ProcessPoolExecutor(1) as executor: 
        for trait in traitChr:
            if DBIsFile(name+'qq',trait+'.png',parms):
                continue
                
            futures.append(executor.submit(qqPlotsHelp,trait,parms))
            
        wait(futures,return_when=ALL_COMPLETED)
    
    return()

def qqPlotsHelp(trait,parms):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']
    
    if trait in snpChr:
        snpData=DBRead(name+'process/snpData',parms,True)
        snps=snpData.groupby('chr').count().reset_index()
        snps['Mbp']=(snps['Mbp']-snps[snps['chr']==trait]['Mbp'].iloc[0]).abs()
        snps=snps[snps.chr!=trait]
        closestChr=snps['chr'].iloc[snps.Mbp.argsort().iloc[0]]
        cisTrait=trait
    else:
        closestChr='chr1'  
        cisTrait='chr1'
    
    trans=[]
    for snp in snpChr:
        print('loading pvals from snp '+snp+' trait '+trait)
        df=2*norm.sf(np.abs(DBRead(name+'score/p-'+snp+'-'+trait,parms,True)))
        if snp==cisTrait:
            cis=df
        if snp==closestChr:
            close=df
        if not (snp==cisTrait or snp==closestChr):
            trans+=[df]
   
    trans=np.concatenate(trans,axis=0)
    
    cis=-np.log10(np.sort(cis.flatten()))
    trans=-np.log10(np.sort(trans.flatten()))
    close=-np.log10(np.sort(close.flatten()))
    
    cisX=-np.log10(np.arange(1,len(cis)+1)/(len(cis)+1))
    transX=-np.log10(np.arange(1,len(trans)+1)/(len(trans)+1))
    closeX=-np.log10(np.arange(1,len(close)+1)/(len(close)+1))
    
    fig,axs=plt.subplots(1,1,dpi=50)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(20,forward=True)

    axs.set_title('trait '+trait)
    axs.set_ylabel('observed distribution')
    axs.set_xlabel('null distribution')
    
    axs.scatter(cisX,cis,label='cis')
    axs.scatter(transX,trans,label='trans')
    axs.scatter(closeX,close,label='closest')
    
    xMin=min(axs.get_xlim()[0], axs.get_ylim()[0])
    xMax=max(axs.get_xlim()[1], axs.get_ylim()[1])
    
    axs.set_xlim([xMin,xMax])
    axs.set_ylim([xMin,xMax])
    axs.legend()
    
    axs.plot(axs.get_xlim(), axs.get_ylim(), ls="--", c=".3")    
        
    if not os.path.exists(local+name+'qqPlots'):
        os.mkdir(local+name+'qqPlots')
        
    fig.savefig(local+name+'qqPlots/'+trait+'.png',bbox_inches='tight')
    plt.close('all')

    DBUpload(name+'qqPlots/'+trait+'.png',parms,False)
    print('wrote qq-'+trait,flush=True)
    
    return()

