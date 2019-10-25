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
from ail.dataPrepPython.genMeans import *
from ail.dataPrepPython.genZScores import *

from ail.statsPython.setupELL import *
from ail.statsPython.genIIDStats import *
from ail.statsPython.genSimStats import *
from ail.statsPython.makePVals import *

from ail.plotPython.plotPVals import *
from ail.plotPython.manhattanPlots import *
from ail.plotPython.qqPlots import *
from ail.plotPython.usThem import *
from ail.plotPython.plotZ import *
from ail.plotPython.minP import *

from ail.opPython.DB import *
from ail.opPython.setupFolders import *
from datetime import datetime

local=os.getcwd()+'/'

ops={
    'makePedFiles':False,
    'genZScore':False,
    'corr':False,
    'qq':False,
    'z2':False,
    'man':False,
    'minP':False,
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
    'linBatch':True,
    'traitChr':[1],
    'numIIDSegments':1000,
    'numSimStatSegments':10
}

parmsAll['comparisonBatch/']={
    'response':'hipExp',
    'quantNormalizeExpr':False,
    'PCIsPreds':False,
    'linBatch':False,
    'traitChr':[1],
    'numIIDSegments':1000,
    'numSimStatSegments':10
}

parmsAll['remPredPCInGemma/']={
    'response':'hipRaw',
    'quantNormalizeExpr':True,
    'PCIsPreds':True,
    'linBatch':False,
    'traitChr':[x for x in range(1,21)],
    'numIIDSegments':1000,
    'numSimStatSegments':10
}

#########################################################################################33

names=list(parmsAll.keys())
names={loc:names[loc] for loc in range(len(names))}
opArgs={loc:opsList[loc] for loc in range(len(opsList))}

args=[int(x) for x in input(str(opArgs)+': ').split(' ')]
name=names[int(input(str(names)+' : '))]
print(name,np.array(opsList)[args],flush=True)

for arg in args:
    ops[opArgs[arg]]=True

ellDSet=[0.1]
ellMeths=['ell-'+str(x) for x in ellDSet]
parms={
    **ops,
    'local':local,
    'name':name,
    'numCores':cpu_count(),
    'numPCs':10,
    'smallNumCores':3,
    'snpChr':[x for x in range(1,20)],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo',
    'allChrGRM':False,
    'cisMean':False,
    'numQSnps':3e9,
    'ellEps':1e-9,
    'ellDSet':ellDSet,
    'binsPerIndex':500,
    'numQSnps':3e9,
    'IIDReps':5000000,
    'Types':ellMeths,
    'firstEntry':True,
    'logName':'ga/log/'+opArgs[args[0]]+'-'+str(datetime.now()),
    'muEpsRange':[],
    **parmsAll[name]
}

setupFolders(parms)
name=parms['name']

if parms['makePedFiles']:
    print('makePedFiles')
    makePedFiles(parms)

# create the actual Z scores by running gemma lmm
if parms['score']:
    print('gen scores')   
    genZScores(parms)

if parms['corr']:
    print('corr')
    genMeans(parms)
    genLZCorr('all',parms)

# qq plots of p-vals
if parms['qq']:
    print('qq plots')
    qqPlots(parms)

# plot histogram of squared Z scores by chromosome
if parms['z2']:
    print('do zvar')   
    plotZ(parms)

if parms['man']:    
    print('MA Plots')
    manhattanPlots(parms,B=10)

if parms['minP']:    
    print('minP Plots')
    minP(parms)

if parms['usThem']:
    print('us Them')
    usThem({**parms,'cpu':10})

if parms['genOffDiag']:
    print('genOffDiag',flush=True)
    genOffDiag(parms) 

if parms['setupELL']:
    setupELL(parms)

if parms['genIIDStats']:
    DBCreateFolder(name,'iid',parms)
    genIIDStats(parms,ellMeths,'all')

if parms['genSimStats']:
    DBCreateFolder(name,'stats',parms)
    genSimStats(parms,ellMeths,'all')
    
if parms['makePVals']:
    makePVals(parms)      

if parms['plotPVals']:
    plotPVals(parms)      
