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
from limix.qc import normalise_covariance
from limix.stats import linear_kinship
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

def writeY(parms):
    local=parms['local']
    numCores=parms['numCores']
    etaSq=parms['parms'][0]
    numSubjects=parms['parms'][1]
    numTraits=parms['parms'][2]
    yParm=parms['yParm']
    ySeed=parms['ySeed']
    
    traitSize=[numSubjects,numTraits]

    np.random.seed(ySeed)
    
    if 'realTraits' in yParm:
        traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)

        robjects=pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
        mouseGenes=robjects['mouseGenes']
        mouseGenes=mouseGenes[(mouseGenes['chrom'].isin([str(x) for x in range(1,22)]))&(mouseGenes['gene_name'].isin(
            traits.columns))]

        traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'].astype(int),
            'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})
        traitData=traitData.sort_values(by=['chr','trait']).iloc[:numTraits]
        traitData.to_csv('Y/traitData',index=False,sep='\t')    
        Y=traits[traitData.trait].values
    else:
        pd.DataFrame({'trait':range(numTraits),'chr':1,'Mbp':0}).to_csv('Y/traitData',index=False,sep='\t')
        if 'depTraits' in yParm:
            traits=getRealTraits(parms)  
            LTraitCorr=makePSD(np.corrcoef(traits,rowvar=False))
        if 'indepTraits' in yParm:
            LTraitCorr=np.eye(numTraits)
        LgrmAll=np.loadtxt('grm/Lgrm-1',delimiter='\t')
        Y=np.sqrt(etaSq)*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LTraitCorr.T)+np.sqrt(1-etaSq)*np.matmul(
            norm.rvs(size=traitSize),LTraitCorr.T)

    if 'quantNorm' in yParm:
        Y=norm.ppf((np.argsort(Y,axis=0)+1)/(numSubjects+1))
    if 'stdNorm' in yParm:
        Y=(Y-np.mean(Y,axis=0))/np.std(Y,axis=0)
    if 'noNorm' in yParm:
        pass

    np.savetxt('Y/Y.phe',np.array([['0',str(int(id_))]+row for id_,row in enumerate(Y.tolist())]),delimiter='\t',fmt='%s')
    np.savetxt('Y/Y.txt',Y,delimiter='\t')    