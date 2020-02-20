import matplotlib
matplotlib.use('agg')
import numpy as np
import os
import pdb
import sys

sys.path=['source']+sys.path

from opPython.setupFolders import *

from simPython.makeSimInputFiles import *
from dataPrepPython.genZScores import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from plotPython.plotZ import *

import subprocess
from scipy.stats import norm

from ELL.ell import *

snpSize=[500]
numSubjects=600

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(snpSize)+1)]
traitSubset=list(range(500))

ctrl={
    'etaSq':0,
    'YType':'simIndep',#['simDep','real','simIndep']
    'snpType':'sim',#['real','sim','random','test']
    'modelTraitIndep':'indep',#['indep','dep']
    'lmm':['gemma','lmm','bed'], #['gemma','fast','lmm','lm','bed','bimbam','ped']
    'grm':'gemmaStd',#['gemmaNoStd','gemmaStd','fast','none']
    'normalize':'none',#['quant','none','std']
    'snpSize':snpSize,
    'numSubjects':numSubjects
}
ops={
    'file':sys.argv[0],
    'ellDSet':ellDSet,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'colors':colors,
    'refReps':1e6,    
    'simLearnType':'Full',
    'response':'hipRaw',
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'traitSubset':traitSubset,
    'maxSnpGen':5000,
    'transOnly':False,
    'numTraits':100
}

#######################################################################################################

parms=setupFolders(ctrl,ops)

DBCreateFolder('diagnostics',parms)
DBCreateFolder('inputs',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

makeSimInputFiles(parms)

genZScores(parms)

#######################################################################################################
pdb.set_trace()
z=np.loadtxt('score/waldStat-1',delimiter='\t')
N=z.shape[1]
plotZ(z)

#######################################################################################################

offDiag=np.array([0]*int(N*(N-1)/2))
stat=ell(np.array(ellDSet),offDiag)

#######################################################################################################

#stat.load()
stat.fit(10*N,1000*N,3000,1e-6) # numLamSteps0,numLamSteps1,numEllSteps,minEll
stat.save()

#######################################################################################################

ell=stat.score(z)
ref=stat.score(norm.rvs(size=[int(parms['refReps']),N]))

#######################################################################################################

monteCarlo=stat.monteCarlo(ref,ell)

#######################################################################################################

markov=stat.markov(ell)

#######################################################################################################

plotPower(monteCarlo,parms,'mc',['mc-'+str(x) for x in ellDSet])
plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet])
#pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
#pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)


DBFinish(parms)