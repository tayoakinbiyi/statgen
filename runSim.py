import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.setupFolders import *
from ail.dataPrepPython.simSetup import *
from ail.simPython.genY import *
from ail.simPython.genSnps import *
from ail.dataPrepPython.score import *
from ail.simPython.genH0ZCorr import *
from ail.plotPython.plotZ import *
from ail.simPython.genH1ZScores import *
from ail.statsPython.setupELL import *
from ail.simPython.genH0PVals import*
from ail.simPython.genH1PVals import*
from ail.simPython.makePower import *
from ail.plotPython.plotPower import *

local=os.getcwd()+'/'

ops={
    'setup':False,
    'genY':False,
    'genSnps':False,
    'genH0ZScores':False,
    'genH1ZScores':False,
    'genH0ZCorr':False,
    'plotH0Z':False,
    'setupELL':False,
    'genH0PVals':False,
    'genH1PVals':False,
    'makePower':False,
    'plotPower':False
}
opsList=list(ops.keys())
opArgs={loc:opsList[loc] for loc in range(len(opsList))}

args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

for arg in args:
    ops[opArgs[arg]]=True

parms={
    'etaGRM':.5,
    'etaError':.5,
    'ellEps':1e-6,
    'delta':.1,
    'ellDSet':[.5,.1],
    'local':local,
    'name':'sim/',
    'cpu':cpu_count(),
    'numPCs':10,
    'smallCpu':3,
    'maxZReps':1000,
    'minPower':250,
    'maxPower':750,
    'grmSnpChr':['chr'+str(x) for x in range(1,20)],
    'snpChr':['chr0','chr1'],
    'traitChr':['chr'+str(x) for x in range(1,3)],
    'muList':[1,2],
    'epsList':[10,20],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'pvalCutOff':.01,
    'numQSnps':3e9,
    'maxZReps':100,
    'treeHeights':[.005,.01,.03,.04,.05,.1,.2,.3,.4,.5,.6,.7],
    'H0SnpSize':40001,
    'H1SnpSize':10001,
    'grmParm':'s',
    'wald':True,
    'allChrGRM':True,
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo',
    'response':'hipRaw',
    'quantNormalizeExpr':True,
    'remPCFromSnp':False,
    'remPCFromTraits':False,
    'remPCCorrSnp':False,
    'PCIsPreds':False,
    'CovIsPreds':False,
    'remCovFromTraits':True,
    'linBatch':False,
    'cisMean':True
}

print(np.array(opsList)[args],flush=True)
setupFolders(parms)

if ops['setup']:
    simSetup(parms)

if ops['genY']:
    genY(parms)

if ops['genSnps']:
    genSnps(parms)

if ops['genH0ZScores']:
    score(parms)

if ops['genH1ZScores']:
    genH1ZScores(parms)
    
if ops['genH0ZCorr']:
    genH0ZCorr(parms)
    
if ops['plotH0Z']:
    plotZ(parms)

if ops['setupELL']:
    for d in ellDSet:
        setupELL(d,parms)
    
if ops['genH0PVals']:
    genH0PVals(parms)

if ops['genH1PVals']:
    genH1PVals(parms)

if ops['makePower']:
    makePower(parms)

if ops['plotPower']:
    plotPower(parms)