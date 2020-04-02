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
    
    Y,QS,M,snps=makeSim(parms,fit=True)
    vY=np.corrcoef(Y,rowvar=False)
    plotCorr(vY,'vY')
    myHist(vY[np.triu_indices(numTraits,1)],'vY-hist')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    wald,eta=runLimix(Y,QS,M,snps)
    plotZ(wald,prefix='wald-')
    np.savetxt('wald',wald,delimiter='\t')
    
    #######################################################################################################
    
    wald=np.loadtxt('wald',delimiter='\t')
    
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
        
    stat=ELL.ell.ell(int(.3*numTraits),numTraits)
    stat.preCompute(1e3,offDiag=offDiag)
    gnull=stat.gnullScore(wald)
    pre=stat.preScore(wald)
    stat.plot(stat.monteCarlo(stat.preScore(zDep),pre),'diagnostics/ellDepMC-Y:{}'.format(parms['yParm']))
    stat.plot(stat.monteCarlo(stat.gnullScore(zDep),gnull),'diagnostics/ellDepMidMC-Y:{}'.format(parms['yParm']))
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
    'pedigreeMult':1
}

ctrl={
    'numSubjects':400,
    'numGrmSnps':5000,
    'numDataSnps':100,
    'numTraits':100,
    'eta':0.3
}

#######################################################################################################

parms={**ctrl,**ops}

setupFolders()
diagnostics()    
log(parms)
    
myMain({**parms,'yParm':'indep'})
#myMain({**parms,'yParm':'dep'})

git('{} : Y, run with dep and indep ell'.format(sys.argv[0][:-3]))