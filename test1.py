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
from plotPython.plotZ import *
from utility import *
from plotPython.myHist import *
from limix.model.lmm import LMM
from limix.stats import linear_kinship
from dataPrepPython.makePedigreeSnps import *
from numpy_sugar.linalg import economic_qs

from statsPython.gbj import *

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
    d=parms['d']
    traitCorrSource=parms['traitCorrSource']
    traitCorrRho=parms['traitCorrRho']
    numCores=cpu_count()
    V_Z=parms['V_Z']
    eta=parms['eta']
    
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
        traitCorr=np.corrcoef(pd.read_csv('../data/'+response+'.txt',sep='\t',index_col=0,header=0).values[
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
    '''
    U,lam,Vt=np.linalg.svd(vZ)
    gamma=numTraits/numSubjects
    phase=(1+np.sqrt(gamma))**2

    lrg=np.where(lam>phase)[0]
    sml=np.where(lam<=phase)[0]
    
    l=(lam[lrg]+1-gamma+np.sqrt((lam[lrg]+1-gamma)**2-4*lam[lrg]))/2
    c=np.sqrt((1-gamma/(l-1)**2)/(1+gamma/(l-1)))
    s=np.sqrt(1-c**2)

    if V_Z=='operator':
        lam[lrg]=l
        lam[sml]=1
    if V_Z=='frobenius':
        lam[lrg]=l*c**2+s**2
        lam[sml]=1
    if V_Z=='stein':
        lam[lrg]=l/(c**2+l*s**2)
        lam[sml]=1
    if V_Z=='frechet':
        lam[lrg]=(s**2+np.sqrt(l)*c**2)**2
        lam[sml]=1
    if V_Z=='simple':
        lam[:]=lam[:]
    if V_Z=='eye': 
        lam[:]=1

    vZ=U@np.diag(lam)@U.T
    makeCov=np.diag(1/np.sqrt(np.diag(vZ)))
    vZ=makeCov@vZ@makeCov
    plotCorr(vZ,'vZorig')
    '''
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    stat=ELL.ell.ell(int(parms['d']*numTraits),numTraits,vZ,numCores=1)
    stat.preCompute(1e3)
    pre=stat.score(wald)
    stat.plot(stat.monteCarlo(pre,1e6,1e5),'diagnostics/ellMC-Y')
    stat.plot(stat.markov(pre),'diagnostics/ellMarkov-Y')        
    stat.plot(gbj('GBJ',wald,numCores=1,offDiag=offDiag),'diagnostics/gbj')
    stat.plot(gbj('GHC',wald,numCores=1,offDiag=offDiag),'diagnostics/ghc')
    

ops={
    'seed':5754,
    'numGrmSnps':10000,
    'd':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':1200,
    'numDataSnps':3,
    'numTraits':8000,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'traitCorrSource':'exchangeable',
    'traitCorrRho':0.2,
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
