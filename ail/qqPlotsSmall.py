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

def qqPlotsSmall(traitChr,scratchDir,chrs): 
    figQQ,axsQQ=plt.subplots(1,1,dpi=50)

    figQQ.set_figwidth(10,forward=True)
    figQQ.set_figheight(10,forward=True)
                    
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
    
    # cis plot
    mMax=max(max(max(-np.log(cisX)),max(-np.log(transX))),max(max(-np.log(cis)),max(-np.log(trans))))*1.1   
    mMin=min(min(min(-np.log(cisX[cis<.01])),min(-np.log(cis[cis<.01]))),min(min(-np.log(transX[trans<.01])),
        min(-np.log(trans[trans<.01]))))*.9
    axsQQ.set_xlim([mMin,mMax])
    axsQQ.set_ylim([mMin,mMax])
    axsQQ.plot(-np.log(cisX[cis<.01]),-np.log(cis[cis<.01]),'ro',label='cis')
    axsQQ.plot(-np.log(transX[trans<.01]),-np.log(trans[trans<.01]),'bo',label='trans')
    axsQQ.set_ylabel(traitChr)
    axsQQ.plot(axsQQ.get_xlim(), axsQQ.get_ylim(), ls="--", c=".3")
    
    figQQ.savefig(scratchDir+'qq_'+traitChr+'.png',bbox_inches='tight')
    plt.close('all')

    
chrs=['chr'+str(x) for x in range(1,20)]
for traitChr in chrs:
    qqPlotsSmall(traitChr,scratchDir,chrs)


