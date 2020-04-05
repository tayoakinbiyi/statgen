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
from dataPrepPython.runH1 import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from plotPython.plotZ import *
from utility import *

from statsPython.gbj import *

from scipy.stats import norm
import ELL.ell
from plotPython.plotCorr import *

from rpy2.robjects.packages import importr

def myMain(parms):
    gbjR=importr('GBJ')
                
    numTraits=parms['numTraits']
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    Y,QS,M,snps,Lgrm=makeSim(parms)
    vY=np.corrcoef(Y,rowvar=False)
    plotCorr(vY,'vY')
    myHist(vY[np.triu_indices(numTraits,1)],'vY-hist')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    wald,eta=runLimix(Y,QS,M,snps)
    plotZ(wald,prefix='wald-')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
        
    vZ=np.corrcoef(wald,rowvar=False)
    L=makePSD(vZ)
    plotCorr(vZ,'vZ')
    offDiag=vZ[np.triu_indices(numTraits,1)]    
    myHist(offDiag,'vZ-hist')

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    zDep=np.matmul(norm.rvs(size=[int(1e5),ctrl['numTraits']]),L.T)
    zIndep=norm.rvs(size=[int(1e5),ctrl['numTraits']])
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
        
    stat=ELL.ell.ell(int(.3*numTraits),numTraits,qList=[0.1,0.05,0.01,0.001])
    stat.preCompute(1e3,offDiag=offDiag)
    gnull=stat.gnullScore(wald)
    pre=stat.preScore(wald)
    stat.plot(stat.monteCarlo(stat.preScore(zDep),pre),'diagnostics/ellDepMC-Y:{}'.format(parms['yParm']))
    #stat.plot(gbj(gbjR.GBJ,wald,offDiag=offDiag),'diagnostics/gbjDep-Y:{}'.format(parms['yParm']))
    stat.plot(stat.markov(pre),'diagnostics/ellDepMarkov-Y:{}'.format(parms['yParm']))
    stat.plot(stat.monteCarlo(stat.gnullScore(zIndep),gnull),'diagnostics/ellIndepMC-Y:{}'.format(parms['yParm']))
    #stat.plot(gbj(gbjR.GBJ,wald),'diagnostics/gbjIndep-Y:{}'.format(parms['yParm']))
    stat.preCompute(1e3)
    stat.plot(stat.markov(gnull),'diagnostics/ellIndepMarkov-Y:{}'.format(parms['yParm']))
    

ops={
    'response':'hipRaw',
    'snpSeed':760,
    'ySeed':0,
    'maxSnpGen':5000,
    'numGrmSnps':10000,
    'eta':0.3,
}

ctrl={
    'numSubjects':1200,
    'numDataSnps':10000,
    'numTraits':1000,
    'pedigreeMult':0.5,
    'yParm':'dep'
}

#######################################################################################################

parms={**ctrl,**ops}

setupFolders()
diagnostics()    
log(parms)
    
myMain(parms)

git('{} mice, {} snps, {} traits, subsample {}, Y is {}'.format(ctrl['numSubjects'],ctrl['numDataSnps'],
    ctrl['numTraits'],ctrl['pedigreeMult'],ctrl['yParm']))
