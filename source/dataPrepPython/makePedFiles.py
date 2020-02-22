from scipy.stats import norm
import pandas as pd
import numpy as np
import pdb
import subprocess

from opPython.DB import *
from dataPrepPython.makeGrm import *
from dataPrepPython.makeTraitPedFiles import *
from dataPrepPython.initSnpTraits import *
from dataPrepPython.writeSnps import *

from multiprocessing import cpu_count

import json

def makePedFiles(parms):
    local=parms['local']
    response=parms['response']
    numCores=parms['numCores']
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']    
    quantNormalizeExpr=parms['quantNormalizeExpr']
    
    predsList=parms['predsList']
    remList=parms['remList']
    
    DBCreateFolder('ped',parms)
    DBCreateFolder('LZCorr',parms)
    DBCreateFolder('grm',parms)
    
    traits,traitData,snps,snpData=initSnpTraits(parms,snps=True)

    ################################################### make covar files ###################################################
    
    covariates=pd.read_csv(local+'data/covars.hipp.208.txt',sep='\t',header=None,index_col=None,
        names=['intercept','sex','batch'])

    preds=covariates[['intercept']]
    
    if sum(pd.Series(predsList+remList,name='list',dtype=str).str.contains('pc'))>0:
        tList=pd.Series(predsList+remList,name='predsList',dtype=str)
        numPCs=tList[tList.str.contains('pc')].str.extract(r'pc-([0-9]+)',expand=True).iloc[0,0]
        U,S,Vt=np.linalg.svd(traits)
        PCs=U[:,0:numPCs]
        
    if sum(pd.Series(predsList,name='predsList',dtype=str).str.contains('pc'))>0:
        preds=pd.concat([preds,pd.DataFrame(PCs)],axis=1)
        
    if 'sex' in predsList:
        preds.insert(preds.shape[1],'sex',covariates['sex'])
        
    if 'linBatch' in predsList:
        preds.insert(preds.shape[1],'linBatch',covariates['batch'])
    
    if 'nonLinBatch' in predsList:
        preds=pd.concat([preds,pd.get_dummies(covariates['batch'],drop_first=True)],axis=1)
        
    pd.concat([pd.DataFrame({'Family ID':range(len(preds)),'Individual ID':0}),preds],axis=1).to_csv(
        'ped/cov.phe',header=False,index=False,sep='\t')
    preds.to_csv('ped/cov.txt',header=False,index=False,sep='\t')
                  
    ################################################### make trait pheno file ###################################################
        
    if len(remList)>0:
        remDF=covariates[['intercept']]
        
        if sum(pd.Series(remList,name='remList').str.contains('pc'))>0:
            remDF=pd.concat([remDF,pd.DataFrame(PCs)],axis=1)
        
        if 'sex' in remList:
            remDF.insert(remDF.shape[1],'sex',covariates['sex'])
            
        if 'linBatch' in remList:
            remDF.insert(remDF.shape[1],'linBatch',covariates['batch'])
        
        if 'nonLinBatch' in remList:
            remDF=pd.concat([remDF,pd.get_dummies(covariates['batch'],drop_first=True)],axis=1)
        
        reg=MultiOutputRegressor(LinearRegression(fit_intercept=False),n_jobs=-1).fit(remDF,traits)
        traits=(traits-reg.predict(remDF).values)    

    makeTraitPedFiles(traits,traitData,parms)

    ################################################### make snp ped ###################################################

    print('load snps',flush=True)
    
    writeSnps(snps,snpData,parms)
    
    ################################################### GRM ###################################################
    
    for snp in snpChr:  
        makeGrm(parms,snp,False)
    
    makeGrm(parms,'all',fastGrm,False)
                            
    return()
