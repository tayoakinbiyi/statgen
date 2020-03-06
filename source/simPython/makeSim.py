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

from dataPrepPython.writeGrm import *
from dataPrepPython.writeSnps import *
from dataPrepPython.writeY import *

def makeSim(parms,genSnps=True,genGrm=True,genY=True,genCov=True):
    print('simSetup')
    
    local=parms['local']
    numCores=parms['numCores']
    response=parms['response']
    muEpsRange=parms['muEpsRange']
    
    snpsSeed=parms['snpsSeed']
    YSeed=parms['YSeed']
    
    etaSq=parms['parms'][0]
    numSubjects=parms['parms'][1]
    numTraits=parms['parms'][2]
    numSnps=parms['parms'][3]
                                  
    if genSnps:
        DBCreateFolder('snps',parms)        
        writeSnps(parms)

    if genGrm:
        DBCreateFolder('grm',parms)
        writeGrm(parms,1)    

        M=len(muEpsRange)
        for snp in range(2,len(numSnps)):
            subprocess.call(['ln','-s','fast-eigen-1', 'grm/fast-eigen-'+str(snp)])
            subprocess.call(['ln','-s','gemma-eigen-1', 'grm/gemma-eigen-'+str(snp)])
            subprocess.call(['ln','-s','limix-1', 'grm/limix-'+str(snp)])
    
    if genY:
        DBCreateFolder('grm',parms)
        writeY(parms)
        traitSize=[numSubjects,numTraits]

        np.random.seed(YSeed)

        if 'realTraits' in yParm:
            Y=getRealTraits(parms)    
        else:
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

    if genCov:
        np.savetxt('cov/cov.phe',np.array([[0,id_,1] for id_ in range(numSubjects)]),delimiter='\t',fmt='%s')
        np.savetxt('cov/cov.txt',np.ones([numSubjects,1]),delimiter='\t',fmt='%s')

    return()
