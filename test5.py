import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb

import sys
import os

sys.path=[os.getcwd()+'/source']+sys.path
from dill.source import getsource
from dataPrepPython.makeSim import *
from dataPrepPython.runLimix import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from plotPython.plotZ import *
from utility import *
from plotPython.myHist import *
from limix.model.lmm import LMM

from statsPython.gbj import *

from scipy.stats import norm
import ELL.ell
from plotPython.plotCorr import *

from rpy2.robjects.packages import importr

def myMain(parms):
    gbjR=importr('GBJ')
                
    numSnps=parms['numDataSnps']
    numTraits=parms['numTraits']

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    Y,QS,M,snps,grm=makeSim(parms)    
    plotCorr(grm,'grm')
    np.savetxt('snps',snps,delimiter='\t')
    np.savetxt('Y',Y,delimiter='\t')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    wald,eta=runLimix(Y,QS,M,snps,0.9999)
    np.savetxt('wald',wald,delimiter='\t')
    myHist(eta,'eta'+str(numTraits))
    myQQ(eta,np.mean(wald**2,axis=0),'eta','wald**2')
    
    wald=np.loadtxt('wald',delimiter='\t')
    
    plotZ(wald,'wald')
        
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    vZ=np.corrcoef(wald,rowvar=False)
    plotCorr(vZ,'vZorig')
    offDiagOrig=vZ[np.triu_indices(vZ.shape[1],1)]    
    myHist(offDiagOrig,'offDiagOrig')
    
    U,lam,Vt=np.linalg.svd(vZ)
    gamma=numTraits/parms['numSubjects']
    phase=(1+np.sqrt(gamma))**2

    lrg=np.where(lam>phase)[0]
    sml=np.where(lam<=phase)[0]
    
    l=(lam[lrg]+1-gamma+np.sqrt((lam[lrg]+1-gamma)**2-4*lam[lrg]))/2
    c=np.sqrt((1-gamma/(l-1)**2)/(1+gamma/(l-1)))
    s=np.sqrt(1-c**2)

    if parms['V(Z)']=='operator':
        lam[lrg]=l
        lam[sml]=1
    if parms['V(Z)']=='frobenius':
        lam[lrg]=l*c**2+s**2
        lam[sml]=1
    if parms['V(Z)']=='stein':
        lam[lrg]=l/(c**2+l*s**2)
        lam[sml]=1
    if parms['V(Z)']=='frechet':
        lam[lrg]=(s**2+np.sqrt(l)*c**2)**2
        lam[sml]=1
    if parms['V(Z)']=='simple':
        lam[:]=lam[:]
    if parms['V(Z)']=='eye': 
        lam[:]=1

    vZ=U@np.diag(lam)@U.T
    makeCov=np.diag(1/np.sqrt(np.diag(vZ)))
    vZ=makeCov@vZ@makeCov
    plotCorr(vZ,'vZorig')

    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]    

    myHist(offDiag,'offDiagNew')
    myQQ(offDiagOrig,offDiag,'orig','new')
    LZ=makeL(vZ)    

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    zRef=np.matmul(norm.rvs(size=[int(1e6),parms['numTraits']]),LZ.T)

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
        
    stat=ELL.ell.ell(int(parms['d']*parms['numTraits']),parms['numTraits'],offDiag)
    #stat.plot(stat.exact(wald),'diagnostics/ellExact-Y:{}'.format(parms['yParm']))
    
    stat.preCompute(1e3)
    pre=stat.score(wald)
    ref=stat.score(zRef)
    stat.plot(stat.monteCarlo(ref,pre),'diagnostics/ellMC-Y')
    stat.plot(stat.markov(pre),'diagnostics/ellMarkov-Y')        
    stat.plot(gbj(gbjR.GBJ,wald,offDiag=offDiag),'diagnostics/gbj')
    stat.plot(gbj(gbjR.GHC,wald,offDiag=offDiag),'diagnostics/ghc')
    

ops={
    'response':'hipRaw',
    'seed':5754,
    'maxSnpGen':5000,
    'numGrmSnps':100,
    'eta':0.3
}

ctrl={
    'numSubjects':1000,
    'numDataSnps':100,
    'numTraits':30,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'd':0.2,
    'traitCorrSource':'empirical',
    'traitCorrRho':0,
    'V(Z)':'simple'
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

diagnostics(parms['seed'])
log(parms)

myMain(parms)

git('{} mice, {} snps, {} traits, subsample {}, Y is {}-{}'.format(parms['numSubjects'],parms['numDataSnps'],
    parms['numTraits'],parms['pedigreeMult'],parms['traitCorrSource'],parms['traitCorrRho']))
