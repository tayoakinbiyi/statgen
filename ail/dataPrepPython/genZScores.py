import pandas as pd
import numpy as np
import subprocess
import pdb
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

from ail.opPython.DB import *

def genZScores(parms):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    numMemGigs=parms['numMemGigs']
    
    traitData=DBRead('process/traitData',parms)

    for trait in traitChr:
        for snp in snpChr:
            if DBIsFile('holds/'+snp+'-'+trait,parms) or DBIsFile('finished',+snp+'-'+trait,parms):
                continue

            DBWrite(np.array([]),'holds/'+snp+'-'+trait,parms,True)

            z=[]
            for k in traitData[traitData['chr']==trait].index:
                cmd=[local+'ext/fastlmmc','-file','ped/ail-'+snp,'-covar','ped/cov.phe','-pheno','ped/ail.phe',
                     '-eigen','ped/eigen-'+snp,'-mpheno',str(k+1),'-out','fastlmm/'+snp+'-'+trait+'-'+str(k+1),
                     '-maxThreads',str(numCores),'-simLearnType','Once']
                subprocess.call(cmd)
                
                df=pd.read_csv('fastlmm/'+snp+'-'+trait+'-'+str(k+1),header=0,index_col=None)
                df.loc[:,'SNP']=df.loc[:,'SNP'].astype(int)
                df=df.sort_values(by='SNP')
                z+=[(df['SnpWeight']/df['SnpWeightSE']).values.flatten()]

            DBWrite(np.concatenate(z,axis=1),'score/'+snp+'-'+trait,parms,True)
        
    notDone=True
    while notDone:
        notDone=False

        for snp in snpChr:
            if notDone:
                continue
            for trait in traitChr:
                if not DBIsFile('finished/'+snp+'-'+trait,parms):
                    notDone=True
                    continue     
                    
        if notDone:
            time.sleep(180)
        
    if not (DBIsFile('holds/LZCorr',parms) or DBIsFile('finished/LZCorr',parms)):
        DBWrite(np.array([]),'holds/LZCorr',parms)

        df=[]
        for trait in traitChr:
            snpDF=[]
            for snp in snpChr:
                snpDF+=[DBRead('score/'+snp+'-'+trait,parms)]
            df+=[np.concatenate(snpDF,axis=1)]
        
        df=np.concatenate(df,axis=0)

        corr=np.corrcoef(df,rowvar=False)

        np.fill_diagonal(corr, 1)

        LZCorr=makePSD(corr)

        print('writing corr',flush=True)
        DBWrite(LZCorr,'LZCorr/LZCorr',parms)

        offDiag=corr[np.triu_indices(len(corr),1)].flatten()

        DBWrite(offDiag,'offDiag/offDiag',parms,True)
        DBWrite(np.array([]),'finished/LZCorr',parms,True)
        
    return()
