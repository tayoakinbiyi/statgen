import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count
import sys

local=os.getcwd()+'/'
sys.path=['source']+sys.path

from opPython.setupFolders import *

from simPython.makeSimPedFiles import *
from simPython.genSimZScores import *
from dataPrepPython.genZScores import *
from dataPrepPython.genLZCorr import *

from statsPython.setupELL import *
from statsPython.genELL import*
from statsPython.makeELLMCPVals_saveZ import*
from statsPython.makeELLMarkovPVals import*
from statsPython.makeGBJPValsSimple import *

from plotPython.plotPower import *
from genPython.makePSD import *

from datetime import datetime
import shutil
import subprocess
import psutil

from ELL.ell import *

warnings.simplefilter("error")

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[100,100,100]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(1000))

parms={
    'file':sys.argv[0],
    'etaGRM':0,
    'etaError':1,
    'ellBetaPpfEps':1e-12,
    'ellKRanLowEps':.1,
    'ellKRanHighEps':1.9,
    'ellKRanNMult':.4,
    'ellDSet':ellDSet,
    'minELLDecForInverse':4,
    'binsPerIndex':500,
    'local':local,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'SnpSize':SnpSize,
    'transOnly':False,
    'colors':colors,
    'refReps':1e6,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'fastlmm':True,
    'grm':2,#[1,2,'fast']
    'traitSubset':traitSubset,
    'numSubjects':208*3,
    'indepTraits':True
}

setupFolders(parms)
'''
DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

DBLog('genZScores')
DBCreateFolder('holds',parms)
genZScores(parms)
'''
N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]

DBLog('genLZCorr')
genLZCorr({**parms,'snpChr':[2]})

LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
zOffDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)]

fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(zOffDiag,bins=60)
fig.savefig('diagnostics/v(z)OffDiag.png')
plt.close('all') 

DBCreateFolder('pvals',parms)

#######################################################################################################

L=np.eye(N)
offDiag=np.matmul(L,L.T)[np.triu_indices(N,1)]
stat=ell(offDiag,N,np.array(ellDSet)*N,reportMem=True)

#######################################################################################################

stat.fit(10,700,1000,12,12) # initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta

#######################################################################################################

zDat=np.concatenate([np.loadtxt('score/waldStat-3-'+str(x),delimiter='\t') for x in traitChr],axis=1)
ell=stat.score(zDat)

#######################################################################################################

zRef=-np.sort(-np.abs(np.matmul(norm.rvs(size=[int(parms['refReps']),N]),L.T)))   
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
