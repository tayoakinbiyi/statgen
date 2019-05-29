import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED

home='/project/abney/'

files={
    'dataDir':home+'ail/data/',
    'scratchDir':home+'ail/scratch/',
    'gemma':home+'ail/gemma'
}

scratchDir=files['scratchDir']

def traitPlot(traitChr,scratchDir):   
    figQQ,axsQQ=plt.subplots(1,2)
    figMA,axsMA=plt.subplots(1,3)

    figQQ.set_figwidth(10,forward=True)
    figQQ.set_figheight(5,forward=True)

    figMA.set_figwidth(30,forward=True)
    figMA.set_figheight(5,forward=True)
    
    axsQQ=axsQQ.flatten()
    axsMA=axsMA.flatten()
                    
    pvals=[]
    for snpChr in ['chr'+str(x) for x in range(1,20)]:
        pvals+=[pd.read_csv(scratchDir+'pvals-final-'+snpChr+'-'+traitChr+'.txt',index_col=[0,1],header=[0,1,2])]
     
    pvals=pd.concat(pvals,axis=0)
    snpIndex=pvals.index.levels[0][pvals.index.labels[0]]
    
    if len(set(snpIndex)&{traitChr})==0:
        return()
    
    cisPVals=pvals.loc[snpIndex==traitChr]
    transPVals=pvals.loc[snpIndex!=traitChr]
    
    cis=np.sort(cisPVals.values.flatten())
    trans=np.sort(transPVals.values.flatten())
    
    cisX=np.arange(1,len(cis)+1)/(len(cis)+1)
    transX=np.arange(1,len(trans)+1)/(len(trans)+1)
    
    # cis plot
    mMax=max(max(max(-np.log(cisX)),max(-np.log(transX))),max(max(-np.log(cis)),max(-np.log(trans))))*1.1
    axsQQ[0].set_xlim([0,mMax])
    axsQQ[0].set_ylim([0,mMax])
    axsQQ[0].plot(-np.log(cisX),-np.log(cis),'ro',label='cis')
    axsQQ[0].plot(-np.log(transX),-np.log(trans),'bo',label='trans')
    axsQQ[0].set_ylabel(traitChr)
    axsQQ[0].legend()
    
    mMin=min(min(min(-np.log(cisX[cis<.01])),min(-np.log(cis[cis<.01]))),min(min(-np.log(transX[trans<.01])),
        min(-np.log(trans[trans<.01]))))*.9
    axsQQ[1].set_xlim([mMin,mMax])
    axsQQ[1].set_ylim([mMin,mMax])
    axsQQ[1].plot(-np.log(cisX[cis<.01]),-np.log(cis[cis<.01]),'ro',label='cis')
    axsQQ[1].plot(-np.log(transX[trans<.01]),-np.log(trans[trans<.01]),'bo',label='trans')
    axsQQ[1].set_ylabel(traitChr)
    
    # find top three traits
    mMin=pvals.min(axis=0)
    wMin=mMin.values.argsort()
    
    xAxis=pvals.index.to_frame()[['chr','Mbp']].reset_index(drop=True)
    xAxis['Mbp']/=1e6
    yAxis=pvals.columns.to_frame()[['chr','Mbp']].reset_index(drop=True)
    yAxis['Mbp']=yAxis['Mbp'].astype(float)
    allAxis=pd.concat([xAxis,yAxis],axis=0)
    
    MbpMax=allAxis.groupby('chr').apply(lambda df: pd.DataFrame({'mid':(df['Mbp'].max()+df['Mbp'].min())/2, 
        'maxMbp':df['Mbp'].max()},index=[0])).sort_values(by='chr').reset_index().drop(columns='level_1')
    MbpMax.insert(MbpMax.shape[1],'addMbp',np.append([0],MbpMax['maxMbp'].values[:-1]))
    
    xAxis=xAxis.merge(MbpMax,on='chr')
    yAxis=yAxis.merge(MbpMax,on='chr')
    
    xAxis['Mbp']=xAxis['Mbp']+xAxis['addMbp']
    yAxis['Mbp']=yAxis['Mbp']+yAxis['addMbp']

    for i in range(3):        
        traitMbp=yAxis['Mbp'].iloc[wMin[i]]
        
        axsMA[i].plot(xAxis['Mbp'],-np.log(pvals.iloc[:,wMin[i]]),'ko')
        axsMA[i].set_xticks(MbpMax['mid']+MbpMax['addMbp'])
        axsMA[i].set_xticklabels(MbpMax['chr'])
        axsMA[i].axvline(x=traitMbp,color='r')
        axsMA[i].set_ylabel(traitChr)        
        for index,mRow in MbpMax.iterrows():
            axsMA[i].axvline(x=mRow['maxMbp']+mRow['addMbp'])  
    
    figQQ.savefig(scratchDir+'qq_'+traitChr+'.png',bbox_inches='tight')
    figMA.savefig(scratchDir+'ma_'+traitChr+'.png',bbox_inches='tight')

    
futures=[]
with ProcessPoolExecutor() as executor: 
    for traitChr in ['chr'+str(x) for x in [6,8,10,12,16,18]]:
        traitPlot(traitChr,scratchDir)



