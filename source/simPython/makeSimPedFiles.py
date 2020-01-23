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
    etaGRM=parms['etaGRM']
    etaError=parms['etaError']
    numCores=parms['numCores']
    response=parms['response']
    quantNormalizeExpr=parms['quantNormalizeExpr']
    muEpsRange=parms['muEpsRange']
    fastGrm=parms['fastGrm']
    numSnpChr=parms['numSnpChr']
    numTraitChr=parms['numTraitChr']  
    eyeTrait=parms['eyeTrait']
    numSubjects=parms['numSubjects']
              
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
    
    mouseIds=np.arange(len(snps))
    #pd.DataFrame({'Family ID':mouseIds,'Individual ID':0,'dummy':range(len(mouseIds))}).to_csv(
    #    'ped/snp.phe',header=False,index=False,sep='\t')
    
    makeGrm(parms,1,fastGrm,True)    
    
    M=len(muEpsRange)
    for snp in range(2,numSnpChr+M*numCores+1):
        os.symlink('eigen-1', 'grm/eigen-'+str(snp))
        os.symlink('gemma-1', 'grm/gemma-'+str(snp))
    
    ################################################### gen Y ###################################################
    
    LgrmAll=np.loadtxt('LZCorr/Lgrm-1',delimiter='\t')
    
    if quantNormalizeExpr:
        traits=norm.ppf((np.argsort(traits,axis=0)+1)/(len(traits)+1))

    traitCorr=np.corrcoef(traits,rowvar=False)
    if eyeTrait:
        LTraitCorr=np.eye(traits.shape[1])
    else:
        LTraitCorr=makePSD(traitCorr)
    
    traitSize=[len(snps),traits.shape[1]]
    
    Y=etaGRM*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+etaError*np.matmul(
        norm.rvs(size=traitSize),LTraitCorr.T)
    
    makeTraitPedFiles(Y,traitData,parms)
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0,'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    np.savetxt('ped/cov.txt',np.array([[1]]*len(mouseIds)),delimiter='\t')
    
    print('finished makeSimPedFiles',flush=True)

    return()
