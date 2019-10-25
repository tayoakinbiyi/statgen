import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.setupFolders import *

from ail.dataPrepPython.makeSimPedFiles import *
from ail.dataPrepPython.genSimZScores import *

from ail.simPython.genY import *
from ail.simPython.genSnps import *
from ail.simPython.genSimZScores import *

from ail.statsPython.setupELL import *
from ail.statsPython.genSimStats import*
from ail.statsPython.makePVals import*
from ail.statsPython.genIIDStats import*

from ail.plotPython.plotPower import *
from ail.plotPython.plotZ import *
from ail.plotPython.plotOffDiag import *

from ail.genPython.makePSD import *

from datetime import datetime

local=os.getcwd()+'/'

parms={
    'makeSimPedFiles':False,
    'genSimZScores':False,
    'genOffDiag':False,
    'diagnostics':False,
    'setupELL':False,
    'genIIDStats':False,
    'genSimStats':False,
    'makePVals':False,
    'plotPower':False,
}
parmsList=list(parms.keys())
opArgs={loc:parmsList[loc] for loc in range(len(parmsList))}

args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

for arg in args:
    parms[opArgs[arg]]=True

ellDSet=[.1]
muRange=[.3,.4,.5]
epsRange=[10,8,6,12]
oldMeths=['hc','bj','gnull','score','cpma']
ellMeths=['ell-'+str(x) for x in ellDSet]
colors=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)][0:len(ellMeths+oldMeths)]

parms={
    **parms,
    'etaGRM':.5,
    'etaError':.5,
    'ellEps':1e-9,
    'ellDSet':ellDSet,
    'binsPerIndex':500,
    'local':local,
    'name':'sim/',
    'numCores':cpu_count(),
    'numPCs':10,
    'smallNumCores':3,
    'snpChr':['1','2'],
    'traitChr':['chr1'],
    'muEpsRange':[[mu,eps] for mu in muRange for eps in epsRange],
    'numDecScore':3,
    'H1SnpSize':40001,
    'H2SnpSize':1000,
    'numH2Segments':100,
    'RPath':'ga/ail/R/',
    'PCIsPreds':False,
    'linBatch':False,
    'cisMean':True,
    'Types':oldMeths+ellMeths,
    'colors':colors,
    'firstEntry':True,
    'logName':'ga/log/'+opArgs[args[0]]+'-'+str(datetime.now()),
    'numIIDSegments':1000,
    'IIDReps':500000,    
    'numSimStatSegments':10
}

print(np.array(parmsList)[args],flush=True)
setupFolders(parms)
name=parms['name']

if parms['makeSimPedFiles']:
    makeSimPedFiles(parms)

if parms['genSimZScores']:
    genSimZScores(parms)
    
parms['muEpsRange']+=[[0,0]]
parms['snpChr']=['chr2']

if parms['diagnostics']:                 
    plotOffDiag(['LTraitCorr','LRawTraitCorr'],parms)
    plotOffDiag({'Lgrm-all','LgrmZ'},parms)
    plotZ(parms)

if parms['setupELL']:
    setupELL(parms)
        
if parms['genIIDStats']:
    DBCreateFolder(name,'iid',parms)
    DBWrite(np.eye(DBRead(name+'process/traitData',parms,True).shape[0]),name+'LZCorr/LZCorr-eye',parms)
    genIIDStats(parms,ellMeths,'all')
    genIIDStats(parms,oldMeths,'eye')

if parms['genSimStats']:
    DBCreateFolder(name,'stats',parms)
    genSimStats(parms,ellMeths,'all')
    genSimStats(parms,oldMeths,'eye')

if parms['makePVals']:
    makePVals(parms)

if parms['plotPower']:
    plotPower(parms)
