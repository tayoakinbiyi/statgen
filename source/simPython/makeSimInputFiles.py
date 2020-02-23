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
from dataPrepPython.writeInputs import *
from simPython.makeSimSnps import *

def makeSimInputFiles(parms):
    print('simSetup')
    
    local=parms['local']
    numCores=parms['numCores']
    response=parms['response']
    muEpsRange=parms['muEpsRange']
    
    sim=parms['sim']
    etaSq=sim[-4]
    numSubjects=sim[-3]
    numTraits=sim[-2]
    numSnps=sim[-1]
              
    DBCreateFolder('inputs',parms)
    DBCreateFolder('geneDrop',parms)
    DBCreateFolder('LZCorr',parms)
    DBCreateFolder('grm',parms)
    DBCreateFolder('output',parms)
        
    ################################################### snps ped ###################################################
        
    np.random.seed(27)
    
    if 'realSnps' in sim:
        bimBamFmt=np.loadtxt(local+'data/snps.txt',delimiter='\t')[:,2:].T
        bimBamFmt=bimBamFmt[np.mod(np.arange(numSubjects),len(bimBamFmt)),random.sample(range(
            bimBamFmt.shape[1]),sum(numSnps))]        
    elif 'pedigreeSnps' in sim:
        bimBamFmt=makeSimSnps(parms)
    elif 'randSnps' in sim:
        bimBamFmt=np.random.choice([0,1,2],numSubjects*sum(numSnps),True,[.25,.5,.25]).reshape(sum(numSnps),-1)
    
    bimBamFmt=np.round(bimBamFmt,0).astype(int)
    
    ################################################### writeInputs ###################################################

    writeInputs(bimBamFmt,parms)
    
    ################################################### grm sim ###################################################

    np.savetxt('inputs/Y.phe',np.array([['0',str(int(id_)),str(int(id_))] for id_ in range(numSubjects)]),delimiter='\t',fmt='%s')
    makeGrm(parms,1,np.array([1]))    

    M=len(muEpsRange)
    for snp in range(2,20+M*numCores+1):
        subprocess.call(['ln','-s','fast-eigen-1', 'grm/fast-eigen-'+str(snp)])
        subprocess.call(['ln','-s','gemma-eigen-1', 'grm/gemma-eigen-'+str(snp)])
    
    ################################################### gen Y ###################################################        
            
    traitSize=[numSubjects,numTraits]
    
    np.random.seed(110)
    
    if 'realTraits' in sim:
        assert numSubjects==len(traits)
        traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

        robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
        mouseGenes=robjects['mouseGenes']
        mouseGenes=mouseGenes[(mouseGenes['chrom'].isin(range(1,22)))&(mouseGenes['gene_name'].isin(traits.columns))]

        traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
            'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
        traitData=traitData.sort_values(by=['chr','trait']).iloc[:numTraits]
        traitData.to_csv('ped/traitData',index=False,sep='\t')    
        traits=traits[traitData.trait].values

        Y=traits    
    else:
        if 'depTraits' in sim:
            LTraitCorr=makePSD(np.corrcoef(traits,rowvar=False))
        if 'indepTraits' in sim:
            LTraitCorr=np.eye(numTraits)
        LgrmAll=np.loadtxt('LZCorr/Lgrm-1',delimiter='\t')
        Y=np.sqrt(etaSq)*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+np.sqrt(1-etaSq)*np.matmul(
            norm.rvs(size=traitSize),LTraitCorr.T)
    
    if 'quantNorm' in sim:
        Y=norm.ppf((np.argsort(Y,axis=0)+1)/(numSubjects+1))
    if 'stdNorm' in sim:
        Y=(Y-np.mean(Y,axis=0))/np.std(Y,axis=0)

    np.savetxt('inputs/Y.phe',np.array([['0',str(int(id_))]+row for id_,row in enumerate(Y.tolist())]),delimiter='\t',fmt='%s')

    ################################################### cov ped ###################################################
        
    np.savetxt('inputs/cov.phe',np.array([[0,id_,1] for id_ in range(numSubjects)]),delimiter='\t',fmt='%s')
    np.savetxt('inputs/cov.txt',np.ones([numSubjects,1]),delimiter='\t',fmt='%s')

    return()
