import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb

import sys
import os

sys.path=[os.getcwd()+'/source']+sys.path
from dill.source import getsource
from dataPrepPython.runLimix import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from utility import *
from plotPython.myHist import *
from plotPython.plotPower import *
from limix.model.lmm import LMM
from limix.stats import linear_kinship
from dataPrepPython.makePedigreeSnps import *
from numpy_sugar.linalg import economic_qs
from statsPython.scoreTest import *
from statsPython.storeyQ import *
from statsPython.minP import *
from statsPython.f import *
from statsPython.gbj import *
from statsPython.cpma import *
from statsPython.monteCarlo import *
from statsPython.markov import *
from functools import partial
from scipy.stats import norm
import ELL.ell
from plotPython.plotCorr import *

from rpy2.robjects.packages import importr

def myMain(parms):
    numDataSnps=parms['numDataSnps']
    numGrmSnps=parms['numGrmSnps']
    numTraits=parms['numTraits']
    numSubjects=parms['numSubjects']
    pedigreeMult=parms['pedigreeMult']
    d=int(parms['d']*numTraits)
    traitCorrSource=parms['traitCorrSource']
    traitCorrRho=parms['traitCorrRho']
    numCores=cpu_count()
    V_Z=parms['V_Z']
    eta=parms['eta']
    refReps=int(1e6)
    maxRefReps=int(1e5)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    miceRange=np.random.choice(208,int(pedigreeMult*208),replace=False)    

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    grm=linear_kinship(makePedigreeSnps(numSubjects,miceRange,numGrmSnps,numCores),verbose=True)    
    QS=economic_qs(grm)
    Lgrm=makeL(grm)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
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
            
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    snps=makePedigreeSnps(numSubjects,miceRange,numDataSnps,numCores)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    wald,eta=runLimix(Y,QS,np.ones([numSubjects,1]),snps,0.9999)
    np.savetxt('wald',wald,delimiter='\t')
    
    #wald=np.loadtxt('wald',delimiter='\t')
        
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    vZ=np.corrcoef(wald,rowvar=False)
    plotCorr(vZ,'vZorig')
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    '''
    stat=ELL.ell.ell(d,vZ,numCores=10)    
    stat.preCompute(1e3)
    func=partial(f,stat.lamEllByK,stat.ellGrid)
    storey=partial(storeyQ,int(vZ.shape[1]*.5))
    
    numCores=10
    plotPower(monteCarlo(cpma,wald,vZ,refReps,maxRefReps,numCores,'cpma'),'diagnostics/cpma')
    plotPower(monteCarlo(func,wald,vZ,refReps,maxRefReps,numCores,'ell'),'diagnostics/ell-Y')      
    plotPower(markov(func,wald,stat.lamEllByK,stat.ellGrid,offDiag,numCores),'diagnostics/ellMarkov-Y')  
    plotPower(monteCarlo(scoreTest,wald,vZ,refReps,maxRefReps,numCores,'scoreTest'),'diagnostics/scoreTest-Y')      
    plotPower(monteCarlo(storey,wald,vZ,refReps,maxRefReps,numCores,'storeyQ'),'diagnostics/storeyQ-Y')      
    plotPower(monteCarlo(minP,wald,vZ,refReps,maxRefReps,numCores,'minP'),'diagnostics/minP-Y')     
    '''
    #stat.plot(gbj('GBJ',wald,numCores=3,offDiag=offDiag),'diagnostics/gbj')
    plotPower(gbj('GHC',wald,numCores=3,offDiag=offDiag),'diagnostics/ghc')
    

ops={
    'seed':None,
    'numGrmSnps':100,
    'd':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':500,
    'numDataSnps':3,
    'numTraits':100,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'traitCorrSource':'empirical',
    'traitCorrRho':1,
    'V_Z':'simple'
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

diagnostics(parms['seed'])
log(parms)

myMain(parms)

git('{} mice, {} snps, {} traits, subsample {}, Y is {}-{}'.format(parms['numSubjects'],parms['numDataSnps'],
    parms['numTraits'],parms['pedigreeMult'],parms['traitCorrSource'],parms['traitCorrRho']))
