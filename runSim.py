import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.setupFolders import *

from ail.simPython.genNullY import *
from ail.dataPrepPython.score import *
from ail.simPython.genNullZCorr import *
from ail.plotPython.plotNullZ import *
from ail.simPython.genH1ZScores import *
from ail.statsPython.setupELL import *
from ail.simPython.genH0PVals import*
from ail.simPython.genH1PVals import*
from ail.simPython.makePower import *
from ail.plotPython.plotPower import *

local=os.getcwd()+'/'

ops={
    'process':False,
    'genNullY':False,
    'genNullZScores':False,
    'genNullZCorr':False,
    'plotNullZ':False,
    'genH0ZScores':False,
    'genH1ZScores':False,
    'setupELL':False,
    'genH0PVals':False,
    'genH1PVals':False,
    'makePower':False,
    'plotPower':False
}
opsList=list(ops.keys())
opArgs={loc:opsList[loc] for loc in range(len(opsList))}

H1Chr='chr'+str(input('H1 chr number : '))
args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

for arg in args:
    ops[opArgs[arg]]=True

parms={
    'H1Chr':H1Chr,
    'etaGRM':.4,
    'etaError':.6,
    'ellEps':1e-6,
    'delta':.1,
    'ellDSet':[.5,.1],
    'local':local,
    'name':'sim-'+H1Chr+'/',
    'cpu':cpu_count(),
    'numPCs':10,
    'smallCpu':3,
    'maxZReps':1000,
    'minPower':250,
    'maxPower':750,
    'snpChr':['chr'+str(x) for x in range(1,6) if 'chr'+str(x)!=H1Chr],
    'traitChr':['chr'+str(x) for x in range(1,3) if 'chr'+str(x)!=H1Chr],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'pvalCutOff':.01,
    'maxZReps':100,
    'treeHeights':[.005,.01,.03,.04,.05,.1,.2,.3,.4,.5,.6,.7],
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

print(H1Chr,np.array(opsList)[args],flush=True)
setupFolders(parms)

if ops['process']:
    process(parms)

if ops['genNullY']:
    genNullY(parms)

if ops['genNullZScores']:
    score(parms)

if ops['genNullZCorr']:
    genNullZCorr(parms)
    
if ops['plotNullZ']:
    plotNullZ(parms)

if ops['genH0ZScores']:
    genH1ZScores(parms)
    
if ops['genH1ZScores']:
    genH1ZScores(parms)
    
if ops['setupELL']:
    for d in ellDSet:
        setupELL(d,parms)
    
if ops['genH0PVals']:
    genNullPVals(parms)

if ops['genH1PVals']:
    genH1PVals(parms)

if ops['makePower']:
    makePower(parms)

if ops['plotPower']:
    plotPower(parms)