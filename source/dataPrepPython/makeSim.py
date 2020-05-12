import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import random
from scipy.stats import norm

from utility import *
from limix.stats import linear_kinship
from dataPrepPython.makePedigreeSnps import *
from numpy_sugar.linalg import economic_qs
from plotPython.plotCorr import *

def makeSim(parms):
    print('makeSim')
        
    numSubjects=parms['numSubjects']
    numGrmSnps=parms['numGrmSnps']
    numDataSnps=parms['numDataSnps']
    numTraits=parms['numTraits']
    eta=parms['eta']
    
    seed=parms['seed']
    
    maxSnpGen=parms['maxSnpGen']
    pedigreeMult=parms['pedigreeMult']
    
    traitCorrSource=parms['traitCorrSource']
    traitCorrRho=parms['traitCorrRho']
        
    np.random.seed(seed)
    '''
    snps={'grm':np.random.uniform(size=[numSubjects,numGrmSnps])*2,
          'data':np.random.uniform(size=[numSubjects,numDataSnps])*2}
    '''
    snps=makePedigreeSnps(numSubjects,numDataSnps,numGrmSnps,maxSnpGen,pedigreeMult)
    
    np.savetxt('data',snps['data'],delimiter='\t')
    np.savetxt('grm',snps['grm'],delimiter='\t')
    #snps={'grm':np.loadtxt('grm',delimiter='\t').astype(int),'data':np.loadtxt('data',delimiter='\t').astype(int)}
    
    grm=linear_kinship(snps['grm'],verbose=True)
    
    QS=economic_qs(grm)
    Lgrm=makeL(grm)
    
    if 'empirical'==traitCorrSource:
        traitCorr=np.corrcoef(pd.read_csv('../data/hipRaw.txt',sep='\t',index_col=0,header=0).values[
            :,0:numTraits],rowvar=False)
    if 'exchangeable'==traitCorrSource:
        traitCorr=np.ones([numTraits,numTraits])
    
    traitCorr=np.eye(numTraits)+traitCorrRho*(traitCorr-np.diag(np.diag(traitCorr)))
        
    LTraitCorr=makeL(traitCorr)
    plotCorr(traitCorr,'traitCorr')
    
    Y=(np.sqrt(eta)*Lgrm @ norm.rvs(size=[numSubjects,numTraits])+np.sqrt(1-eta)*norm.rvs(size=
        [numSubjects,numTraits]))@LTraitCorr.T
        
    M=np.ones([numSubjects,1])
        
    return(Y,QS,M,snps['data'].reshape(len(snps['data']),-1),grm)

def isFloat(num):
    try:
        float(num)
    except ValueError:  # String is not a number
        return(False)
    return(True)