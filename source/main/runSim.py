import matplotlib
matplotlib.use('agg')
import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.setupFolders import *

from ail.simPython.makeSimPedFiles import *
from ail.simPython.genSimZScores import *
from ail.dataPrepPython.genZScores import *
from ail.dataPrepPython.genLZCorr import *

from ail.statsPython.setupELL import *
from ail.statsPython.makeELLMarkovPVals import*

from ail.statsPython.genELL import*
from ail.statsPython.makeELLMCPVals import*
from ail.statsPython.makeGBJPVals import *

from ail.plotPython.plotPower import *
from ail.plotPython.plotZ import *
from ail.plotPython.plotOffDiag import *
from ail.plotPython.plotGrm import *
from ail.plotPython.plotNullRef import *

from ail.genPython.makePSD import *


from datetime import datetime

local=os.getcwd()+'/'

parms={
    'makeSimPedFiles':False,
    'genZScores':False,
    'diagnostics':False,
    'genLZCorr':False,
    'genSimZScores':False,
    'setupELL':False,
    'genELL':False,
    'makeELLMCPVals':False,
    'makeELLMarkovPVals':False,
    'makeGBJPVals':False,
    'plotPower':False
}
parmsList=list(parms.keys())
opArgs={loc:parmsList[loc] for loc in range(len(parmsList))}

args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

for arg in args:
    parms[opArgs[arg]]=True

ellDSet=[.1,.5]
muRange=[.2,.3]
epsRange=[8,10]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[40000,10000,500]

parms={
    **parms,
    'LSnp':2,
    'etaGRM':.5,
    'etaError':.5,
    'ellEps':1e-12,
    'ellDSet':ellDSet,
    'minELLDecForInverse':4,
    'binsPerIndex':1000,
    'local':local,
    'name':'sim/',
    'numCores':cpu_count(),
    'numServers':3,
    'numPCs':10,
    'snpChr':[snp for snp in range(1,len(SnpSize)+1)],
    'traitChr':[20],
    'muEpsRange':[[mu,eps] for mu in muRange for eps in epsRange],
    'numDecScore':3,
    'SnpSize':SnpSize,
    'transOnly':False,
    'colors':colors,
    'logName':'log/'+opArgs[args[0]]+'-'+str(datetime.now()),
    'RefReps':1000000,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'snpFile':'ail.genos.ATGC.gwasSNPs.txt',
    'quantNormalizeExpr':False,
    'verbose':True
}

print(np.array(parmsList)[args],flush=True)
setupFolders(parms,opArgs[args[0]])
name=parms['name']

if parms['makeSimPedFiles']:
    makeSimPedFiles(parms,fastGrm=False)

if parms['genZScores']:
    DBCreateFolder('score',parms)
    genZScores(parms,fastlmm=True)
    
if parms['genLZCorr']:
    genLZCorr(parms,[2])

if parms['diagnostics']:       
    #plotGrm(parms)
    plotZ(parms,'simSnps',[3])

if parms['genSimZScores']:
    genSimZScores(parms,fastlmm=True)
    
if parms['setupELL']:
    setupELL(parms)
        
snpChr=[str(len(SnpSize)+x) for x in range(len(parms['muEpsRange'])+1)]

if parms['genELL']:
    genELL(parms,snpChr)
    
if parms['makeELLMCPVals']:
    makeELLMCPVals(parms,snpChr)
    
if parms['makeELLMarkovPVals']:
    makeELLMarkovPVals(parms,snpChr)

if parms['makeGBJPVals']:
    makeGBJPVals(parms,[2,3])
    
if parms['plotPower']:
    plotPower(parms,[3])
