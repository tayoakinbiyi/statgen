import pandas as pd
import numpy as np
import subprocess
import pdb
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import time
from functools import partial
import datetime

from ail.opPython.DB import *
from ail.genPython.makePSD import *

def genZScores(parms):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    simLearnType=parms['simLearnType']
    cisMean=parms['cisMean']
    
    DBLog('genZScores',parms)
            
    traitData=pd.read_csv('ped/traitData',sep='\t',header=0,index_col=None)
    snpData=pd.read_csv('ped/snpData',sep='\t',header=0,index_col=None)
    
    for trait in traitChr:
        for snp in snpChr:
            nameParm=snp+'-'+trait

            if os.path.exists('holds/genZScores-'+nameParm) or os.path.exists('score/'+nameParm):
                continue

            np.savetxt('holds/genZScores-'+nameParm,np.array([]),delimiter='\t')

            z=[]
            M=sum(traitData['chr']==trait)
            numLeft=M
            with ProcessPoolExecutor(numCores) as executor:
                k=0
                while numLeft>0:
                    futures=[]
                    for core in range(min(numLeft,numCores)):
                        print(nameParm+'-'+str(k+1)+' of '+str(M)+'\t'+str(datetime.now()),flush=True)
                        DBLog(nameParm+'-'+str(k+1)+' of '+str(M),parms)
                        cmd=[local+'ext/fastlmmc','-file','ped/ail-'+snp,'-covar','ped/cov.phe','-pheno','ped/ail.phe',
                            '-eigen','ped/eigen-'+snp,'-mpheno',str(k+1),'-out','fastlmm/'+nameParm+'-'+str(k+1),
                             '-maxThreads',str(1),'-simLearnType',simLearnType]
                        futures+=[executor.submit(subprocess.call,cmd)]
                        k+=1
                        numLeft-=1
                    
                    wait(futures,return_when=ALL_COMPLETED)
            
            for k in range(M):
                df=pd.read_csv('fastlmm/'+nameParm+'-'+str(k+1),header=0,index_col=None,sep='\t')
                df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
                df=df.sort_values(by='SNP')
                z+=[(df['SNPWeight']/df['SNPWeightSE']).values.reshape(-1,1)]

            np.savetxt('score/'+nameParm,np.concatenate(z,axis=1),delimiter='\t')
            
    notDone=True
    while notDone:
        notDone=False

        for snp in snpChr:
            if notDone:
                continue
            for trait in traitChr:
                if not os.path.exists('score/'+snp+'-'+trait):
                    print('waiting for '+snp+'-'+trait+'\t'+str(datetime.now()),flush=True)
                    notDone=True
                    continue     
                    
        if notDone:
            time.sleep(60)
    
    if not (os.path.exists('holds/LZCorr') or os.path.exists('LZCorr/LZCorr')):
        np.savetxt('holds/LZCorr',np.array([]),delimiter='\t')
        
        snpLoc=snpData['chr'].values
        traitLoc=traitData['chr'].values
        
        traitDF=pd.DataFrame(index=range(len(snpLoc)),columns=range(len(traitLoc)),dtype=float)        

        for trait in traitChr:
            for snp in snpChr:
                if trait==snp and (not cisMean):
                    continue
                traitDF.loc[snpLoc==snp,traitLoc==trait]=DBRead('score/'+snp+'-'+trait,parms)

        corr=traitDF.corr().values
        LZCorr=makePSD(corr)

        print('writing corr',flush=True)
        np.savetxt('LZCorr/LZCorr',LZCorr,delimiter='\t')
        
        np.savetxt('LZCorr/Leye',np.eye(LZCorr.shape[0]),delimiter='\t')
        
    return()


