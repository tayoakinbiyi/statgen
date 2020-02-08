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
    SnpSize=parms['SnpSize']
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
    
    assert snpType in ['real','sim']
    
    if snpType=='real':
        snps=pd.read_csv(local+'data/snps.txt',sep='\t',header=0,index_col=None)
        snps=snps.iloc[:,2:].T.values
    else:
        subprocess.call(['cp',local+'ext/sampleIds.txt','geneDrop/sampleIds.txt'])
        pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
            'geneDrop/parms.txt',index=False,header=None)

    snpList=[]
    t_num=0
    while t_num<numSubjects:
        if snpType=='sim':
            snpList+=[makeSimSnps(parms).values[np.min(208,numSubjects-t_num),:]]
        else:
            snpList+=[snps[np.arange(min(208,numSubjects-t_num))][:,random.sample(range(snps.shape[1]),np.sum(SnpSize))]]        
        t_num+=len(snpList[-1])

    snps=pd.DataFrame(np.concatenate(snpList,axis=0))

    snpData=pd.DataFrame({'chr':[snp for snp,size in zip(snpChr,SnpSize) for ind in range(size)],
                          'ID':range(np.sum(SnpSize)),'genetic dist': 0,'Mbp':range(np.sum(SnpSize))})
                
    writeSnps(snps,snpData,parms)

    ################################################### grm sim ###################################################
        
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
    
    assert YType in ['simDep','real','simIndep']
    if YType =='simDep':
        LTraitCorr=makePSD(traits)
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
    if normalize=='std':
        Y=(Y-np.mean(Y,axis=0))/np.std(Y,axis=0)
    
    makeTraitPedFiles(Y,traitData,parms)
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0,'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    np.savetxt('ped/cov.txt',np.array([[1]]*len(snps)),delimiter='\t')
    
    print('finished makeSimPedFiles',flush=True)

    return()
