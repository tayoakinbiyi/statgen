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

import subprocess
from scipy.stats import norm

from ELL.ell import *


ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[10000,10000,1000]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(1000))

ctrl={
    'etaSq':0.5,
    'numSubjects':208*3,
    'YTraitIndep':True,
    'modelTraitIndep':True,
    'fastlmm':False
}
ops={
    'file':sys.argv[0],
    'ellDSet':ellDSet,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'SnpSize':SnpSize,
    'colors':colors,
    'refReps':1e6,    
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'grm':'gemmaStd',#['gemmaNoStd','gemmaStd','fast']
    'traitSubset':traitSubset,
    'maxSnpGen':5000,
    'transOnly':False
}

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
N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]

DBLog('genLZCorr')
genLZCorr({**parms,'snpChr':[2]})
LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
zOffDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)]
'''
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(zOffDiag,bins=60)
fig.savefig('diagnostics/v(z)OffDiag.png')
plt.close('all') 

DBCreateFolder('pvals',parms)
'''
#######################################################################################################

offDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)]
stat=ell(N,np.array(ellDSet)*N,parms['numCores'],True)

#######################################################################################################

stat.fit(20,1000,2000,8,1e-7,offDiag) # initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta

#######################################################################################################

zDat=np.concatenate([np.loadtxt('score/waldStat-3-'+str(x),delimiter='\t') for x in traitChr],axis=1)
ell=stat.score(zDat)

#######################################################################################################

#zRef=-np.sort(-np.abs(np.matmul(norm.rvs(size=[int(parms['refReps']),N]),LZCorr.T)))  
zRef=np.concatenate([np.loadtxt('score/waldStat-2-'+str(x),delimiter='\t') for x in traitChr],axis=1)
ref=stat.score(zRef)

#######################################################################################################

monteCarlo=stat.monteCarlo(ref,ell)

#######################################################################################################

markov=stat.markov(ell)

#######################################################################################################

pvals=pd.DataFrame(-np.log10(np.sort(np.concatenate([monteCarlo,markov],axis=1),axis=0)),
    columns=[nm+'-'+str(x) for nm in ['mc','markov'] for x in ellDSet])
plotPower(pvals,parms)
pvals.quantile([.05,.01],axis=0).to_csv('diagnostics/exact.csv',index=False)

DBFinish(parms)
