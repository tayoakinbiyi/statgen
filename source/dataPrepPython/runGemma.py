import pandas as pd
import numpy as np
import subprocess
import pdb
import os
from limix.model.lmm import LMM
from ELL.util import *
from multiprocessing import cpu_count
from utility import *
import warnings
import statsmodels.api as sm
from scipy.stats import norm,t

def runGemma(numSnps,numSubjects,numTraits,lmm):
    b_waldStat=bufCreate('waldStat',[numSnps,numTraits])
    b_waldStat[0][:]=np.nan
    
    b_eta=bufCreate('eta',[numSnps,numTraits])
    b_eta[0][:]=np.nan

    pids=[]
    numCores=cpu_count()
    for core in range(numCores):
        traitRange=np.arange(core*int(np.ceil(numTraits/numCores)),min(numTraits,(core+1)*int(np.ceil(numTraits/numCores))))
        if len(traitRange)==0:
            continue
        #runLimixHelp(Y,QS,M,snps,b_waldStat,b_eta,b_reml,b_fail,traitRange,numSu)
        #b_eta,b_reml,b_fail,
        runGemmaHelp(core,lmm,b_waldStat,b_eta,traitRange,numSubjects)
        pids+=[remote(runGemmaHelp,core,lmm,b_waldStat,b_eta,traitRange,numSubjects)]

    for pid in pids:
        os.waitpid(0, 0)
    
    return(bufClose(b_waldStat),bufClose(b_eta))

def runGemmaHelp(core,lmm,b_waldStat,b_eta,traitRange,numSubjects):
    count=1
    #pdb.set_trace()
    cmd=['../ext/gemma','-o','gemma-'+str(core),'-c','M','-p','Y','-g','snps']
    if lmm:
        cmd+=['-lmm','4','-d','D','-u','U','-km','1']
    else:
        cmd+=['-lm','1'] 
        
    for trait in traitRange:
        print('{} : {} of {}'.format('runGemma',count,len(traitRange)),flush=True)

        loopCmd=cmd+['-n',str(trait+1)]
        subprocess.call(loopCmd)
        df=pd.read_csv('output/gemma-'+str(core)+'.assoc.txt',header=0,index_col=None,sep='\t')
        pdb.set_trace()
        if lmm:
            b_eta[0][:,trait]=(df['l_remle']/(1+df['l_remle']))
        b_waldStat[0][:,trait]=norm.ppf(t.cdf(df['beta']/df['se'],numSubjects-2))
            
        count+=1
        
    return()