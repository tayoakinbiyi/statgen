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

def runFast(numSnps,numSubjects,numTraits):
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
        #runFastHelp(core,b_waldStat,b_eta,traitRange,numSubjects)
        pids+=[remote(runFastHelp,core,b_waldStat,b_eta,traitRange,numSubjects)]

    for pid in pids:
        os.waitpid(0, 0)
    
    return(bufClose(b_waldStat),bufClose(b_eta))

def runFastHelp(core,b_waldStat,b_eta,traitRange,numSubjects):
    count=1
        
    for trait in traitRange:
        print('runFast : {} of {}'.format(count,len(traitRange)),flush=True)

        cmd=['../ext/fastlmmc','-file','data','-pheno','Y.phe',#'-eigen','eigen',
             '-mpheno',str(trait+1),'-out','fastlmm-'+str(core),'-maxThreads','1',
             '-simLearnType','Full','-ML','-verboseOut','-linreg']
        subprocess.call(cmd)
        df=pd.read_csv('fastlmm-'+str(core),header=0,index_col=None,sep='\t')

        df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
        df=df.sort_values(by='SNP')
        #pdb.set_trace()
        ''''
        b_eta[0][:,trait]=df['GeneticVar']/(df['ResidualVar']+df['GeneticVar'])
        b_waldStat[0][:,trait]=df['SNPWeight']/df['SNPWeightSE']
        '''
        b_eta[0][:,trait]=0
        b_waldStat[0][:,trait]=df['SnpWeight']/df['SnpWeightSE']
            
        count+=1
        
    return()