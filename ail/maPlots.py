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

def maPlot(traitChr,scratchDir,chrs): 
    B=5
    figMA,axsMA=plt.subplots(1,B,dpi=50)

    figMA.set_figwidth(50,forward=True)
    figMA.set_figheight(5,forward=True)
    
    axsMA=axsMA.flatten()
                    
    pvals=[]
    for snpChr in chrs:
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
    
    # find top three traits
    wMin=pvals.min(axis=0).values.flatten().argsort()
    
    xAxis=pvals.index.to_frame()[['chr','Mbp']].reset_index(drop=True)
    xAxis['Mbp']/=1e6
    yAxis=pvals.columns.to_frame()[['chr','Mbp']].reset_index(drop=True)
    yAxis['Mbp']=yAxis['Mbp'].astype(float)
    allAxis=pd.concat([xAxis,yAxis],axis=0)
    
    MbpMax=allAxis.groupby('chr').apply(lambda df: pd.DataFrame({'mid':(df['Mbp'].max()+df['Mbp'].min())/2, 
        'maxMbp':df['Mbp'].max()},index=[0])).reset_index().drop(columns='level_1')
    MbpMax.insert(MbpMax.shape[1],'num',MbpMax.chr.str.slice(3).astype(float))
    MbpMax=MbpMax.sort_values(by='num')
    MbpMax.insert(MbpMax.shape[1],'addMbp',np.append([0],MbpMax['maxMbp'].cumsum().iloc[:-1]))
    print(MbpMax)
    
    xAxis=xAxis.merge(MbpMax,on='chr')
    yAxis=yAxis.merge(MbpMax,on='chr')
    
    xAxis['Mbp']=xAxis['Mbp']+xAxis['addMbp']
    yAxis['Mbp']=yAxis['Mbp']+yAxis['addMbp']

    for i in range(B):        
        traitMbp=yAxis['Mbp'].iloc[wMin[i]]
        
        axsMA[i].plot(xAxis['Mbp'],-np.log(pvals.iloc[:,wMin[i]]),'ko')
        axsMA[i].set_xticks(MbpMax['mid']+MbpMax['addMbp'])
        axsMA[i].set_xticklabels(MbpMax['chr'],rotation='vertical')
        axsMA[i].set_ylabel(traitChr)        
        for index,mRow in MbpMax.iterrows():
            axsMA[i].axvline(x=mRow['maxMbp']+mRow['addMbp'])  
        axsMA[i].axvline(x=traitMbp,color='r')
    
    figMA.savefig(scratchDir+'ma_'+traitChr+'.png',bbox_inches='tight')
    plt.close('all')

    
chrs=['chr'+str(x) for x in range(1,20)]
for traitChr in chrs:
    maPlot(traitChr,scratchDir,chrs)


