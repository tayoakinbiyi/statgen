import matplotlib
matplotlib.use('agg')
import numpy as np
import os
import pdb
import sys

sys.path=['source']+sys.path

from opPython.setupFolders import *

from simPython.makeSimPedFiles import *
from dataPrepPython.genZScores import *
from dataPrepPython.genLZCorr import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from plotPython.plotZ import *

import subprocess
from scipy.stats import norm

from ELL.ell import *

numSubjects=10000
snpSize=[200]

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(snpSize)+1)]
traitSubset=list(range(500))

ctrl={
    'etaSq':0,
    'numSubjects':numSubjects,
    'YType':'simIndep',#['simDep','real','simIndep']
    'snpType':'random',#['real','sim','random']
    'modelTraitIndep':'indep',#['indep','dep']
    'lmm':'gemma-lm', #['gemma-lmm','gemma-lm','fastlmm']
    'grm':'none',#['gemmaNoStd','gemmaStd','fast']
    'normalize':'quant',#['quant','none','std']
    'snpSize':snpSize
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
    'transOnly':False
}

#######################################################################################################

parms=setupFolders(ctrl,ops)
'''
DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

DBLog('genZScores')

genZScores(parms)
'''
#######################################################################################################

for snp in parms['snpChr']:
    plotZ(np.concatenate([np.loadtxt('score/waldStat-'+str(snp)+'-'+str(trait),delimiter='\t') for trait in traitChr],axis=1))

#######################################################################################################

'''
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(np.corrcoef(Y,rowvar=False)[np.triu_indices(N,1)],bins=60)
fig.savefig('diagnostics/offDiag-Y.png')
plt.close('all') 

#######################################################################################################

offDiag=np.array([0]*int(N*(N-1)/2))
stat=ell(np.array(ellDSet),offDiag)

#######################################################################################################

stat.load()
if stat.N!=N or np.sum(stat.offDiag)!=np.sum(offDiag):
    stat.fit(10*N,1000*N,3000,1e-6) # numLamSteps0,numLamSteps1,numEllSteps,minEll
    
#######################################################################################################

ell3=stat.score(zDat3)
ell2=stat.score(zDat2)
ell1=stat.score(zDat1)
ellI=stat.score(zDatI)
ref=stat.score(zRef)

#######################################################################################################

monteCarlo3=stat.monteCarlo(ref,ell3)
monteCarlo2=stat.monteCarlo(ref,ell2)
monteCarlo1=stat.monteCarlo(ref,ell1)
monteCarloI=stat.monteCarlo(ref,ellI)

#######################################################################################################

markov3=stat.markov(ell3)
markov2=stat.markov(ell2)
markov1=stat.markov(ell1)

#######################################################################################################

plotPower(monteCarloI,parms,'mcI',['mcI-'+str(x) for x in ellDSet])
plotPower(monteCarlo3,parms,'mc3',['mc3-'+str(x) for x in ellDSet])
plotPower(monteCarlo2,parms,'mc2',['mc2-'+str(x) for x in ellDSet])
plotPower(monteCarlo1,parms,'mc1',['mc1-'+str(x) for x in ellDSet])
plotPower(markov3,parms,'markov3',['markov3-'+str(x) for x in ellDSet])
plotPower(markov2,parms,'markov2',['markov2-'+str(x) for x in ellDSet])
plotPower(markov1,parms,'markov1',['markov1-'+str(x) for x in ellDSet])
#pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
#pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)
'''

DBFinish(parms)