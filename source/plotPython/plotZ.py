import pandas as pd
import numpy as np
import pdb
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, t
from opPython.DB import *
import scipy.stats
import statsmodels.api as sm
from dataPrepPython.genSnpMeans import *
from functools import partial

def plotZ(parms,title,snpChr=None,transCis='all'):
    name=parms['name']
    local=parms['local']
    
    plt.rcParams.update({'font.size': 20})
    
    if snpChr is None:
        snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    
    snpData=pd.read_csv('ped/snpData',sep='\t',index_col=None)
    snpData=snpData[snpData['chr'].isin(snpChr)]
    traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None)
    
    col={'cis':'b','trans':'r'}

    snpLoc=snpData['chr'].values
    traitLoc=traitData['chr'].values
    data=np.full([len(snpLoc),len(traitLoc)],np.nan)

    for trait in traitChr:
        for snp in snpChr:
            print('loading snp '+str(snp)+' trait '+str(trait),flush=True)
            xLoc=np.arange(len(snpLoc))[snpLoc==snp].reshape(-1,1)
            yLoc=np.arange(len(traitLoc))[traitLoc==trait].reshape(1,-1)
            data[xLoc,yLoc]=np.loadtxt('score/waldStat-'+str(snp)+'-'+str(trait),delimiter='\t')
            
    N,numPreds=pd.read_csv('ped/cov.phe',index_col=None,header=None,sep='\t').shape
    numPreds-=2
    
    data=pd.DataFrame(data,columns=traitData['trait'])
    data.insert(0,'snp',range(len(snpData)))
    data.insert(0,'snpChr',snpData['chr'])
    data=pd.melt(data,id_vars=['snp','snpChr'],value_vars=traitData['trait'],var_name='trait',value_name='z')
    data.insert(0,'traitChr',traitData['chr'].values.flatten().tolist()*len(snpData))
    data.loc[:,'z']=data.loc[:,'z'].astype(float)
    data.insert(0,'tZ',norm.ppf(t.cdf(data['z'],N-numPreds)))
    
    if transCis=='cis':
        data=data[data['snpChr']==data['traitChr']]
    elif transCis=='trans':
        data[data['snpChr']!=data['traitChr']]
    
    fig,axs=plt.subplots(3,1,dpi=50,tight_layout=True)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(50,forward=True)

    y=np.sort(data['z'].values.flatten())
    x=norm.ppf(np.arange(1,len(y)+1)/(1+len(y)))
    myQQ(x,y,'all Z-N('+str(np.round(np.mean(y),3))+' , '+str(np.round(np.std(y),3))+')',axs[0],True)

    y=np.sort(data.groupby('snp')['z'].mean())
    x=norm.ppf(np.arange(1,len(y)+1)/(1+len(y)))
    myQQ(x,y,'1st moment (one pt per snp)-N('+str(np.round(np.mean(y),3))+' , '+str(np.round(np.std(y),3))+')',axs[1],True)
    
    #pdb.set_trace()
    y=np.sort(data.groupby('trait')['z'].apply(lambda df:np.sum(df**2)))
    x=chi2.ppf(np.arange(1,len(y)+1)/(1+len(y)),len(snpData))
    myQQ(x,y,'z^2 (one pt per trait)',axs[2],False)
    
    pd.DataFrame({'Type':['Z','Z^2','Z^3','z^4'],'Value':[data['z'].mean(),(data['z']**2).mean(),(data['z']**3).mean(),
        (data['z']**4).mean()],'tValue':[data['z'].mean(),(data['tZ']**2).mean(),(data['tZ']**3).mean(),
        (data['tZ']**4).mean()]}).to_csv('diagnostics/'+title+'.tsv',index=False)
    
    fig.savefig('diagnostics/'+title+'.png')
    plt.close('all')
    
    print('finished plot Z',flush=True)
    return()

def myQQ(x,y,title,axs,std=True):
    if std:
        y=(y-np.mean(y))/np.std(y)

    labMax=max(max(x),max(y))
    labMin=min(min(x),min(y))
    labMax+=.1*(labMax-labMin)
    labMin-=.1*(labMax-labMin)

    axs.set_xlim([labMin,labMax])
    axs.set_ylim([labMin,labMax])
    axs.scatter(x,y,c='.1')
    axs.plot([labMin,labMax], [labMin,labMax], ls="--", c=".3")  
    axs.set_xlabel('theoretical')
    axs.set_ylabel('observed')
    axs.set_title(title)
    
    return()
    