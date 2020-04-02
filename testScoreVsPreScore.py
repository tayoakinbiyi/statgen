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

stat=ELL.ell.ell(int(.3*parms['numTraits']),parms['numTraits'])
'''
z=norm.rvs(size=[int(1e5),parms['numTraits']])
score=[]
for parms['numSubjects'] in [500,1200,1800]:
    for parms['pedigreeMult'] in [.1,.5,1]:
        #Y,QS,M,snps=makeSim(parms,fit=True)
        #wald,eta=runLimix(Y,QS,M,snps)
        #score+=[stat.monteCarlo(stat.gnullScore(z),stat.gnullScore(wald)).reshape(-1,1)]
score=np.concatenate(score,axis=1)
np.savetxt('score',score,delimiter='\t')
'''
score=np.loadtxt('score',delimiter='\t')
count=0
for parms['numSubjects'] in [500,1200,1800]:
    for parms['pedigreeMult'] in [.1,.5,1]:
        cols=['{} - {}'.format(parms['numSubjects'],parms['pedigreeMult'])]
        stat.plot(score[:,count:count+1],'diagnostics/ell-'+cols[0],columns=cols,qList=[.01,.001])
        count+=1

git('{}'.format(sys.argv[0][:-3]))
