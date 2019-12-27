import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.dataPrepPython.makePedFiles import *
from ail.dataPrepPython.genLZCorr import *
from ail.dataPrepPython.genZScores import *

from ail.statsPython.setupELL import *
from ail.statsPython.genRefStats import *
from ail.statsPython.genSimStats import *
from ail.statsPython.makePVals import *

from ail.plotPython.plotPVals import *
from ail.plotPython.manhattanPlots import *
from ail.plotPython.usThem import *
from ail.plotPython.plotZ import *

from ail.opPython.DB import *
from ail.opPython.setupFolders import *

local=os.getcwd()+'/'

ops={
    'makePedFiles':False,
    'genZScores':False,
    'genLZCorr':False,
    'plotZ':False,
    'man':False,
    'usThem':False,
    'setupELL':False,
    'genIIDStats':False,
    'genSimStats':False,
    'makePVals':False,
    'plotPVals':False
}
opsList=list(ops.keys())
opArgs={loc:opsList[loc] for loc in range(len(opsList))}

parmsAll={}

parmsAll['comparison/']={
    'response':'hipExp',
    'quantNormalizeExpr':False,
    'PCIsPreds':False,
    'linBatch':True
}

parmsAll['comparisonBatch/']={
    'response':'hipExp',
    'quantNormalizeExpr':False,
    'PCIsPreds':False,
    'linBatch':False
}

parmsAll['remPredPCInGemma/']={
    'response':'hipRaw',
    'quantNormalizeExpr':True,
    'PCIsPreds':True,
    'linBatch':False
}

#########################################################################################33

names=list(parmsAll.keys())
names={loc:names[loc] for loc in range(len(names))}
opArgs={loc:opsList[loc] for loc in range(len(opsList))}

args=[int(x) for x in input(str(opArgs)+': ').split(' ')]
name=names[int(input(str(names)+' : '))]
numCores=input('number of cores ("" for cpu_count()) :')
numCores=int(numCores) if len(numCores)>0 else cpu_count()
print(name,np.array(opsList)[args],flush=True)

for arg in args:
    ops[opArgs[arg]]=True

ellDSet=[0.1]
ellMeths=['ell-'+str(x) for x in ellDSet]
parms={
    **ops,
    'local':local,
    'name':name,
    'numCores':numCores,
    'numPCs':10,
    'smallNumCores':3,
    'snpChr':[str(x) for x in range(1,10)],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'allChrGRM':False,
    'cisMean':False,
    'numQSnps':3e9,
    'ellEps':1e-9,
    'ellDSet':ellDSet,
    'binsPerIndex':500,
    'numQSnps':3e9,
    'IIDReps':5000000,
    'Types':ellMeths,
    'muEpsRange':[],
    'op':opArgs[args[0]],
    'numIIDSegments':1000,
    'numSimStatSegments':10,
    'traitChr':['1'],
    'simLearnType':'Full',#'Once',
    **parmsAll[name]
}

setupFolders(parms,opArgs[args[0]])
name=parms['name']

if parms['makePedFiles']:
    print('makePedFiles')
    makePedFiles(parms)

# create the actual Z scores by running gemma lmm
if parms['genZScores']:
    print('genZScores')   
    genZScores(parms)

# plot histogram of squared Z scores by chromosome
if parms['plotZ']:
    print('plotZ')   
    plotZ(parms)

if parms['man']:    
    print('MA Plots')
    manhattanPlots(parms,B=10)

if parms['usThem']:
    print('us Them')
    usThem({**parms,'cpu':10})

if parms['setupELL']:
    setupELL(parms)

if parms['genIIDStats']:
    DBCreateFolder('iid',parms)
    genIIDStats(parms,ellMeths,'all')

if parms['genSimStats']:
    DBCreateFolder('stats',parms)
    genSimStats(parms,ellMeths,'all')
    
if parms['makePVals']:
    makePVals(parms)      

if parms['plotPVals']:
    plotPVals(parms)      
