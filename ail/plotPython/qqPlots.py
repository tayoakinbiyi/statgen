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
    
    futures=[]
    #qqPlotsHelp('chr1',parms)
    with ProcessPoolExecutor(smallCpu) as executor: 
        for trait in traitChr:
            futures.append(executor.submit(qqPlotsHelp,trait,parms))
            
        wait(futures,return_when=ALL_COMPLETED)
    
    fig,axs=plt.subplots(len(traitChr),1,dpi=50)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(190,forward=True)

    for i in range(len(traitChr)):
        trait=traitChr[i]
        df=pd.read_csv(scratchDir+'qq-'+trait+'.csv').set_index(keys='observed',drop=True)
        
        axs[i].set_xlim([df.expected.min(),df.expected.max()])
        axs[i].set_ylim([df.index.min(),df.index.max()])
        df.groupby('Type')['expected'].plot(ax=axs[i],legend=True,linewidth=3)
        
        axs[i].set_ylabel(trait+' observed distribution')
        axs[i].set_xlabel('null distribution')
        axs[i].plot(axs[i].get_xlim(), axs[i].get_ylim(), ls="--", c=".3")    
        
    fig.savefig(plotsDir+'qqPlots.png',bbox_inches='tight')
    plt.close('all')
    
def qqPlotsHelp(trait,parms):
    scratchDir=parms['scratchDir']
    plotsDir=parms['plotsDir']
    snpChr=parms['snpChr']
    
    if os.path.isfile(scratchDir+'qq-'+trait+'.csv'):
        return()
    
    if trait in snpChr:
        snpData=pd.read_csv(scratchDir+'snpData.csv')
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
        df=np.loadtxt(scratchDir+'p-'+snp+'-'+trait+'.csv',delimiter=',')
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
    
    observed=np.concatenate([cisX,transX,closeX])
    expected=np.concatenate([cis,trans,close])
        
    df=pd.DataFrame({'Type':['cis']*len(cis)+['trans']*len(trans)+['close']*len(close),
        'expected':expected,'observed':observed}).to_csv(scratchDir+'qq-'+trait+'.csv',index=False)
    print('wrote qq-'+trait+'.csv',flush=True)

