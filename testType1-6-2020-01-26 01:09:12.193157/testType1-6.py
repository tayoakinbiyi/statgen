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
from statsPython.makeELLMCPVals import*
from statsPython.makeELLMarkovPVals import*
from statsPython.makeGBJPValsSimple import *

from plotPython.plotPower import *
from genPython.makePSD import *

from datetime import datetime
import shutil
import subprocess
import psutil
from distutils.dir_util import copy_tree

warnings.simplefilter("error")

local=os.getcwd()+'/'

ellDSet=[.25]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[10000,10000,800]
traitChr=[18]
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
    'numPCs':10,
    'snpChr':snpChr,
    'traitChr':traitChr,
    'SnpSize':SnpSize,
    'transOnly':False,
    'colors':colors,
    'RefReps':100000,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'fastlmm':True,
    'grm':2,
    'gemmaGrmType':2,
    'eyeTrait':False,
    'traitSubset':traitSubset,
    'numSubjects':208*3
}

setupFolders(parms)

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

DBLog('genZScores')
DBCreateFolder('holds',parms)
genZScores(parms)

DBCreateFolder('diagnostics',parms)

DBLog('genLZCorr')

genLZCorr({**parms,'snpChr':[2]})

DBLog('setupELL')
DBCreateFolder('ELL',parms)
setupELL(parms)

DBLog('genELL')
genELL(parms)

DBLog('makeELLMCPVals')
DBCreateFolder('pvals',parms)
makeELLMCPVals(parms)

makeELLMarkovPVals(parms)

DBLog('plotPower')
plotPower(parms)

DBFinish(parms)
