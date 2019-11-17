import pandas as pd
import numpy as np
import pdb
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from ail.opPython.DB import *
import scipy.stats
import statsmodels.api as sm
from ail.dataPrepPython.genSnpMeans import *
from functools import partial

def plotZ(parms,title,snpChr=None,transCis='all'):
    name=parms['name']
    local=parms['local']
    
    plt.rcParams.update({'font.size': 20})
    
    if snpChr is None:
        snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    
    snpData=DBRead('ped/snpData',parms)
    snpData[snpData['chr'].isin(snpChr)]
    traitData=DBRead('ped/traitData',parms)
    
    col={'cis':'b','trans':'r'}
    
    data=pd.DataFrame(columns=traitData['trait'],index=range(len(snpData)))
    
    for trait in traitChr:
        for snp in snpChr:
            print('loading snp '+snp+' trait '+trait,flush=True)
            data.loc[snpData['chr'].values==snp,traitData['chr'].values==trait]=DBRead('score/'+snp+'-'+trait,parms)
    
    data.insert(0,'snp',range(len(snpData)))
    data.insert(0,'snpChr',snpData['chr'])
    data=pd.melt(data,id_vars=['snp','snpChr'],value_vars=traitData['trait'],var_name='trait',value_name='z')
    data.insert(0,'traitChr',traitData['chr'].values.flatten().tolist()*len(snpData))
    data.loc[:,'z']=data.loc[:,'z'].astype(float)
    
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
    myQQ(x,y,'1st moment-N('+str(np.round(np.mean(y),3))+' , '+str(np.round(np.std(y),3))+')',axs[1],True)
    
    #pdb.set_trace()
    y=np.sort(data.groupby('trait')['z'].apply(lambda df:np.sum(df**2)))
    x=chi2.ppf(np.arange(1,len(y)+1)/(1+len(y)),len(snpData))
    myQQ(x,y,'z^2 (one pt per trait)',axs[2],False)
    
    pd.DataFrame({'Type':['Z','Z^2','Z^3','z^4'],'Value':[data['z'].mean(),(data['z']**2).mean(),(data['z']**3).mean(),
        (data['z']**4).mean()]}).to_csv('diagnostics/'+title+'.tsv',index=False)
    
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
    