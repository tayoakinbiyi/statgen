import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import shutil
import random
from scipy.stats import norm

from opPython.DB import *
from genPython.makePSD import *

from dataPrepPython.makeGrm import *
from dataPrepPython.makeTraitPedFiles import *
from dataPrepPython.initSnpTraits import *
from dataPrepPython.writeSnps import *
from simPython.makeSimSnps import *

def makeSimPedFiles(parms):
    print('simSetup')
    
    local=parms['local']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    SnpSize=parms['SnpSize']
    etaSq=parms['etaSq']
    numCores=parms['numCores']
    response=parms['response']
    muEpsRange=parms['muEpsRange']
    numSnpChr=parms['numSnpChr']
    numTraitChr=parms['numTraitChr']  
    numSubjects=parms['numSubjects']
    
    YTraitIndep=parms['YTraitIndep']
              
    DBCreateFolder('ped',parms)
    DBCreateFolder('geneDrop',parms)
    DBCreateFolder('LZCorr',parms)
    DBCreateFolder('grm',parms)
    DBCreateFolder('output',parms)
    
    ################################################### from runAnalysis ###################################################
    
    traits,traitData,snps,snpData=initSnpTraits(parms,snps=False)
    
    ################################################### snps ped ###################################################
    
    subprocess.call(['cp',local+'ext/sampleIds.txt','geneDrop/sampleIds.txt'])

    pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
        'geneDrop/parms.txt',index=False,header=None)

    snps=[]
    t_num=0
    while t_num<numSubjects:
        snps+=[makeSimSnps(parms).values]
        t_num+=len(snps[-1])
   
    snps=pd.DataFrame(np.concatenate(snps,axis=0))
    
    snpData=pd.DataFrame({'chr':[snp for snp,size in zip(snpChr,SnpSize) for ind in range(size)],
        'ID':range(np.sum(SnpSize)),'genetic dist': 0,'Mbp':range(np.sum(SnpSize))})
    
    writeSnps(snps,snpData,parms)

    ################################################### grm sim ###################################################
        
    makeGrm(parms,1,np.array([1]))    
    
    M=len(muEpsRange)
    for snp in range(2,numSnpChr+M*numCores+1):
        os.symlink('eigen-1', 'grm/eigen-'+str(snp))
        os.symlink('gemma-1', 'grm/gemma-'+str(snp))
    
    ################################################### gen Y ###################################################
    
    LgrmAll=np.loadtxt('LZCorr/Lgrm-1',delimiter='\t')
    
    traitSize=[len(snps),traits.shape[1]]
    
    if YTraitIndep in ['indep','dep']:
        if YTraitIndep=='indep':
            LTraitCorr=np.eye(traits.shape[1])
        if YTraitIndep=='dep':
            LTraitCorr=np.loadtxt('LZCorr/LTraitCorr',delimiter='\t')
        Y=np.sqrt(etaSq)*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+np.sqrt(1-etaSq)*np.matmul(
            norm.rvs(size=traitSize),LTraitCorr.T)
    else:
        Y=traits    
    
    makeTraitPedFiles(Y,traitData,parms)
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0,'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    np.savetxt('ped/cov.txt',np.array([[1]]*len(snps)),delimiter='\t')
    
    print('finished makeSimPedFiles',flush=True)

    return()
