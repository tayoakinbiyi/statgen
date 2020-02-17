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
from dataPrepPython.writeSnps import *
from simPython.makeSimSnps import *

def makeSimPedFiles(parms):
    print('simSetup')
    
    local=parms['local']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    snpSize=parms['snpSize']
    etaSq=parms['etaSq']
    numCores=parms['numCores']
    response=parms['response']
    muEpsRange=parms['muEpsRange']
    numSubjects=parms['numSubjects']
    normalize=parms['normalize']
    
    YType=parms['YType']
    snpType=parms['snpType']
              
    DBCreateFolder('ped',parms)
    DBCreateFolder('geneDrop',parms)
    DBCreateFolder('LZCorr',parms)
    DBCreateFolder('grm',parms)
    DBCreateFolder('output',parms)
        
    ################################################### snps ped ###################################################
    
    assert snpType in ['real','sim','random','test']
    
    numSnps=np.sum(snpSize)
    
    np.random.seed(27)
    
    if snpType=='real':
        snps=np.loadtxt(local+'data/snps.txt',delimiter='\t')[:,2:].T
        snps=snps[np.mod(np.arange(numSubjects),len(snps)),random.sample(range(snps.shape[1]),numSnps)]        
    elif snpType=='sim':
        snps=makeSimSnps(parms)
    elif snpType=='random':
        snps=np.random.choice(['A','G'],2*numSubjects*numSnps,True).reshape(numSnps,-1)
        snps=np.char.join(' ',np.char.add(snps[:,0:numSubjects],snps[:,numSubjects:])).T
    elif snpType=='test':
        snps=np.concatenate([np.array(['A A']*int(.5*numSubjects*numSnps)).reshape(-1,numSnps),
                            np.array(['G G']*int(.5*numSubjects*numSnps)).reshape(-1,numSnps)],axis=0)
    
    snps=pd.DataFrame(snps)
    
    snpData=pd.DataFrame({'chr':[snp for snp,size in zip(snpChr,snpSize) for ind in range(size)],
                          'ID':range(numSnps),'genetic dist': 0,'Mbp':range(numSnps)})
                
    writeSnps(snps,snpData,parms)

    ################################################### grm sim ###################################################

    assert parms['grm'] in ['gemmaNoStd','gemmaStd','fast']
    makeGrm(parms,1,np.array([1]))    

    M=len(muEpsRange)
    for snp in range(2,20+M*numCores+1):
        os.symlink('fast-eigen-1', 'grm/fast-eigen-'+str(snp))
        os.symlink('gemma-eigen-1', 'grm/gemma-eigen-'+str(snp))
    
    ################################################### gen Y ###################################################
        
    traitSubset=parms['traitSubset'] if parms['traitSubset'] is not None else range(2000)
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']
    mouseGenes=mouseGenes[mouseGenes['chrom'].isin([str(x) for x in traitChr])]

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
    traitData=traitData.loc[traitData['trait'].isin(traits.columns)]
    traitData=traitData.groupby('chr').apply(lambda df: pd.concat([df.reset_index(drop=True),
        pd.DataFrame({'traitSubset':range(len(df))})],axis=1)).reset_index(drop=True)
    traitData=traitData.sort_values(by=['chr','traitSubset'])
    traitData=traitData[traitData['traitSubset'].isin(traitSubset)]
    traitData.insert(0,'loc',range(len(traitData)))
    traitDataMat=[]
    for trait in traitChr:
        traitDataMat+=[traitData[traitData['chr']==trait]]
    traitData=pd.concat(traitDataMat,axis=0)
    
    traitData.to_csv('ped/traitData',index=False,sep='\t')
    
    traits=traits[traitData.trait].values
        
    traitSize=[len(snps),traits.shape[1]]
    
    np.random.seed(110)
    
    assert YType in ['simDep','real','simIndep']
    if YType =='simDep':
        LTraitCorr=makePSD(np.corrcoef(traits,rowvar=False))
        LgrmAll=np.loadtxt('LZCorr/Lgrm-1',delimiter='\t')
        Y=np.sqrt(etaSq)*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+np.sqrt(1-etaSq)*np.matmul(
            norm.rvs(size=traitSize),LTraitCorr.T)
    elif YType =='real':
        Y=traits    
    else:
        Y=norm.rvs(size=traitSize)        
    
    assert normalize in ['quant','none','std']
    if normalize=='quant':
        Y=norm.ppf((np.argsort(Y,axis=0)+1)/(len(Y)+1))
    elif normalize=='std':
        Y=(Y-np.mean(Y,axis=0))/np.std(Y,axis=0)
    
    makeTraitPedFiles(Y,traitData,parms)
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0,'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    np.savetxt('ped/cov.txt',np.array([[1]]*len(snps)),delimiter='\t')
    
    print('finished makeSimPedFiles',flush=True)

    return()
