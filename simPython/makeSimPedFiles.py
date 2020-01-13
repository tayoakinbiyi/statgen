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

def makeSimPedFiles(parms):
    print('simSetup')
    
    maxSnpGen=parms['maxSnpGen']
    local=parms['local']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    SnpSize=parms['SnpSize']
    maxSnpGen=parms['maxSnpGen']
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
              
    DBCreateFolder('ped',parms)
    DBCreateFolder('geneDrop',parms)
    DBCreateFolder('LZCorr',parms)
    DBCreateFolder('grm',parms)
    DBCreateFolder('output',parms)
    
    ################################################### from runAnalysis ###################################################
    
    traits,traitData,snps,snpData,mouseIds=initSnpTraits(parms,snps=False)
    
    ################################################### snps ped ###################################################
    
    pd.DataFrame({'id':[str(Id)+'.1' for Id in mouseIds]}).to_csv('geneDrop/sampleIds.txt',index=False,header=False)

    pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
        'geneDrop/parms.txt',index=False,header=None)

    snps=[]
    numSnps=0
    size=np.sum(SnpSize)
    while numSnps<size:
        newAdd=min(maxSnpGen,size-numSnps)
        pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
            'geneDrop/map.txt',sep='\t',index=False)
        cmd=[local+'ext/gdrop','-p','geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o','geneDrop/geneDrop']
        subprocess.run(cmd)

        val=pd.read_csv('geneDrop/geneDrop.geno_true',header=0,sep='\t').iloc[:,4:]

        valMAF=np.concatenate([(col.str.split(' ',expand=True).astype(int)>2).sum(axis=1).values.reshape(-1,1) for ind,
            col in val.iteritems()],axis=1)
        val=pd.concat([col.str.split(' ',expand=True).replace({'1':'A','2':'A','3':'G','4':'G'}).apply(lambda x:' '.join(x),axis=1)
            for ind,col in val.iteritems()],axis=1)

        maf=np.mean(valMAF,axis=1)/2
        maf=np.minimum(maf,1-maf)
        val=val.loc[maf>.1,:]

        numSnps+=val.shape[0]
        print('removed '+str(newAdd-val.shape[0])+' snps',flush=True)

        snps+=[val]
    
    snps=pd.concat(snps,axis=0).T
    pdb.set_trace()
        
    snpData=pd.DataFrame({'chr':[snp for snp,size in zip(snpChr,SnpSize) for ind in range(size)],
        'ID':range(np.sum(SnpSize)),'genetic dist': 0,'Mbp':range(np.sum(SnpSize))})
    
    writeSnps(snps,snpData,mouseIds,parms)

    ################################################### grm sim ###################################################
    
    pd.DataFrame({'Family ID':mouseIds,'Individual ID':0,'dummy':range(len(mouseIds))}).to_csv(
        'ped/snp.phe',header=False,index=False,sep='\t')
    
    makeGrm(parms,1,mouseIds,fastGrm,True)    
    
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
        LTraitCorr=np.eye(len(traits))
    else:
        LTraitCorr=makePSD(traitCorr)
    
    traitSize=traits.shape
    
    Y=etaGRM*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+etaError*np.matmul(
        norm.rvs(size=traitSize),LTraitCorr.T)
    
    makeTraitPedFiles(Y,traitData,mouseIds,parms)
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':mouseIds,'Individual ID':0,'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    np.savetxt('ped/cov.txt',np.array([[1]]*len(mouseIds)),delimiter='\t')
    
    print('finished makeSimPedFiles',flush=True)

    return()
