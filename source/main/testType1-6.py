import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count
import sys

sys.path[0]=sys.path[0][:-5]

from opPython.setupFolders import *

from simPython.makeSimPedFiles import *
from simPython.genSimZScores import *
from dataPrepPython.genZScores import *
from dataPrepPython.genLZCorr import *

from statsPython.setupELL import *
from statsPython.genELL import*
from statsPython.makeELLMCPVals_saveZ import*

from plotPython.plotPower import *
from genPython.makePSD import *

from datetime import datetime
import shutil
import subprocess
import psutil
from distutils.dir_util import copy_tree

warnings.simplefilter("error")

local=os.getcwd()+'/'

ellDSet=[.1,.2,.3,.4,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[4000,4000,300]
traitChr=[18]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(1000))

parms={
    'file':sys.argv[0],
    'etaGRM':.5,
    'etaError':.5,
    'ellBetaPpfEps':1e-12,
    'ellKRanLowEps':.1,
    'ellKRanHighEps':1.9,
    'ellKRanNMult':.4,
    'ellDSet':ellDSet,
    'minELLDecForInverse':4,
    'binsPerIndex':700,
    'local':local,
    'numCores':cpu_count(),
    'numPCs':10,
    'snpChr':snpChr,
    'traitChr':traitChr,
    'SnpSize':SnpSize,
    'transOnly':False,
    'colors':colors,
    'RefReps':1000000,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'fastlmm':True,
    'fastGrm':False,
    'eyeTrait':True,
    'traitSubset':traitSubset,
    'numSubjects':208*3
}

parms['name']=parms['file']+'-eye:'+str(parms['eyeTrait'])+'-'+str(ellDSet)

setupFolders(parms,'testType1')

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]
#parms['muEpsRange']=[[.25,int(N*.01)]]

DBLog('genZScores')
DBCreateFolder('holds',parms)
genZScores(parms)

#DBLog('genSimZScores')
#genSimZScores(parms)   

DBLog('genLZCorr')
#genLZCorr({**parms,'snpChr':[2]})
np.savetxt('LZCorr/LZCorr',np.eye(N),delimiter='\t')

DBLog('setupELL')
DBCreateFolder('ELL',parms)
setupELL(parms)

DBLog('genELL')
genELL(parms)

DBLog('makeELLMCPVals')
makeELLMCPVals(parms)

mc=np.sort(np.loadtxt('score/mc',delimiter='\t').flatten())
fast=np.sort(np.loadtxt('score/waldStat-'+str(3)+'-'+str(traitChr[0]),delmiter='\t').flatten())

DBCreateFolder('diagnostics',parms)

fig, axs = plt.subplots(1,1,dpi=50)   
fig.set_figwidth(10,forward=True)
fig.set_figheight(10*len(muEpsRange),forward=True)

axs.scatter(fast,np.arange(1,len(fast)+1)/len(fast),label='fast')
axs.scatter(mc,np.arange(1,len(mc)+1)/len(mc),label='mc')
axs.legend()
fig.savefig('diagnostics/z-compare.png')
    
DBFinish(parms)