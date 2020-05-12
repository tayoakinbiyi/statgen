import numpy as np
import pdb
from scipy.stats import norm
import pandas as pd
import os

from ELL.score.remotes import *
from ELL.util import *
from ELL.F import *
from utility import *

import time

def score(self,testStats,verbose=True):
    t0=time.time()
    if verbose:
        memory('score')
    
    numRows=len(testStats)

    numCores=self.numCores  
    d=self.d
    
    lamEllByK=self.lamEllByK
    testStats=np.sort(2*norm.sf(np.abs(testStats)))[:,0:d]
    
    b_check=bufCreate('check',[3,d])
    
    t1=time.time()
            
    pids=[]
    b_score={}
    for core in range(numCores):
        kRange=np.arange(core*int(np.ceil(d/numCores)),min(d,(core+1)*int(np.ceil(d/numCores))))
        if len(kRange)==0:
            continue
        b_score[core]=bufCreate('pvals-'+str(core),[testStats.shape[0],len(kRange)])
        b_score[core][0][:]=testStats[:,kRange]
        
        pids+=[remote(preScoreHelp,kRange,b_score[core],lamEllByK,b_check)]

    for pid in pids:
        os.waitpid(0, 0)
        
    check=b_check[0][:,np.min(b_check[0][1:],axis=0)>0]
    if verbose:
        print(pd.DataFrame(check[1:],columns=check[0],index=['below','above']),flush=True)
    
    testStats=np.concatenate([b_score[core][0] for core in range(len(b_score))],axis=1)
    
    ellStats=np.min(testStats[:,0:d],axis=1).flatten().astype(int)
    
    ellGrid=self.ellGrid

    for core in b_score:
        bufClose(b_score[core])
    bufClose(b_check)    

    t2=time.time()

    if verbose:
        memory('score')
        log('{} : {} snps, {} min/snp'.format('score',numRows,((t1-t0)+numCores*(t2-t1))/(60*numRows)))
        
    return(ellGrid[ellStats])

def longScore(self,testStats,verbose=True):
    t0=time.time()
    if verbose:
        memory('longScore')
    
    testStats=np.sort(2*norm.sf(np.abs(testStats)))[:,0:d]
    numRows=testStats.shape[0]
               
    pids=[]
    b_score={}
    for core in range(numCores):
        rowRange=np.arange(core*int(np.ceil(numRows/numCores)),min(numRows,(core+1)*int(np.ceil(numRows/numCores))))
        if len(rowRange)==0:
            continue
        b_score[core]=bufCreate('pvals-'+str(core),[len(rowRange),d])
        b_score[core][0][:]=testStats[rowRange,:]
        
        pids+=[remote(scoreHelp,b_score[core],self.N,self.nCr,self.offDiagMeans)]

    for pid in pids:
        os.waitpid(0, 0)
       
    testStats=np.concatenate([b_score[core][0] for core in range(len(b_score))],axis=0)    
    ellStats=np.min(testStats[:,0:d],axis=1)
    
    for core in b_score:
        bufClose(b_score[core])
    
    t1=time.time()

    if verbose:
        memory('longScore')
        log('{} : {} snps, {} min/snp'.format('longScore',numRows,(t1-t0)/(60*numRows)))

    return(ellStats)
    
    



