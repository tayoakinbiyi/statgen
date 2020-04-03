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

ops={
    'response':'hipRaw',
    'snpSeed':0,
    'ySeed':0,
    'maxSnpGen':5000,
    'yParm':'indep'
}

ctrl={
    'numGrmSnps':10000,
    'numDataSnps':10000,
    'numTraits':1000,
    'eta':0.3
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()
diagnostics()    
log(parms)

subList=[2000,3000,4000]
mulList=[.1,.5,1]

stat=ELL.ell.ell(int(.3*parms['numTraits']),parms['numTraits'])

z=norm.rvs(size=[int(1e5),parms['numTraits']])
mc=stat.gnullScore(z)

for parms['numSubjects'] in subList:
    for parms['pedigreeMult'] in mulList:
        Y,QS,M,snps=makeSim(parms,fit=True)
        wald,eta=runLimix(Y,QS,M,snps)
        col=['{} - {}'.format(parms['numSubjects'],parms['pedigreeMult'])]
        stat.plot(stat.monteCarlo(mc,stat.gnullScore(wald)),'diagnostics/ell-'+col[0],columns=col,qList=[.01,.001])

git('{}'.format(sys.argv[0][:-3]))
