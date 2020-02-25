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
from dataPrepPython.writeInputs import *
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
    maxTraitPerChr=parms['maxTraitPerChr']
    
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
        bimBamFmt=np.loadtxt(local+'data/snps.txt',delimiter='\t')[:,2:].T
        bimBamFmt=bimBamFmt[np.mod(np.arange(numSubjects),len(bimBamFmt)),random.sample(range(bimBamFmt.shape[1]),numSnps)]        
    elif snpType=='sim':
        bimBamFmt=makeSimSnps(parms)
    elif snpType=='random':
        bimBamFmt=np.random.choice(['A','G'],2*numSubjects*numSnps,True).reshape(numSnps,-1)
        bimBamFmt=np.char.join(' ',np.char.add(snps[:,0:numSubjects],bimBamFmt[:,numSubjects:])).T
    elif snpType=='test':
        bimBamFmt=np.concatenate([np.array(['A A']*int(.5*numSubjects*numSnps)).reshape(-1,numSnps),
            np.array(['G G']*int(.5*numSubjects*numSnps)).reshape(-1,numSnps)],axis=0)
    pdb.set_trace()
    ################################################### gen Y ###################################################        
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']
    mouseGenes=mouseGenes[(mouseGenes['chrom'].isin([str(x) for x in traitChr]))&(mouseGenes['gene_name'].isin(traits.columns))]

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
    traitData=traitData.groupby('chr').apply(lambda df: pd.concat([df.reset_index(drop=True),
        pd.DataFrame({'traitSubset':range(len(df))})],axis=1)).reset_index(drop=True)
    traitData=traitData.merge(pd.DataFrame({'chr':traitChr,'chrLoc':range(len(traitChr))},index=range(len(traitChr))),
        on='chr').sort_values(by=['chrLoc','traitSubset']).drop(columns='chrLoc')
    traitData=traitData[traitData['traitSubset']<maxTraitPerChr]
    traitData.insert(0,'loc',range(len(traitData)))
    
    traitData.to_csv('ped/traitData',index=False,sep='\t')
    
    traits=traits[traitData.trait].values
        
    traitSize=[numSubjects,traits.shape[1]]
    
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

    ################################################### writeInputs ###################################################

    writeInputs(bimBamFmt,Y,parms)
    
    ################################################### grm sim ###################################################

    assert parms['grm'] in ['gemmaNoStd','gemmaStd','fast']
    makeGrm(parms,1,np.array([1]))    

    M=len(muEpsRange)
    for snp in range(2,20+M*numCores+1):
        subprocess.call(['ln','-s','fast-eigen-1', 'grm/fast-eigen-'+str(snp)])
        subprocess.call(['ln','-s','gemma-eigen-1', 'grm/gemma-eigen-'+str(snp)])
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':range(len(Y)),'Individual ID':0,'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    np.savetxt('ped/cov.txt',np.array([[1]]*len(snps)),delimiter='\t')
    
    print('finished makeSimPedFiles',flush=True)

    return()
