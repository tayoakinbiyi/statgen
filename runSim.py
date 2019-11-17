import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.setupFolders import *

from ail.simPython.makeSimPedFiles import *
from ail.simPython.genSimZScores import *

from ail.statsPython.setupELL import *
from ail.statsPython.genSimStats import*
from ail.statsPython.makePVals import*
from ail.statsPython.genRefStats import*

from ail.plotPython.plotPower import *
from ail.plotPython.plotZ import *
from ail.plotPython.plotOffDiag import *

from ail.genPython.makePSD import *

from datetime import datetime

local=os.getcwd()+'/'

parms={
    'makeSimPedFiles':False,
    'genSimZScores':False,
    'diagnostics':False,
    'setupELL':False,
    'genRefStats':False,
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
muRange=[.3,.4]
epsRange=[10,8]
oldMeths=['noCorr','cpma']
ellMeths=['ell-'+str(x) for x in ellDSet]
colors=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)][0:len(ellMeths+oldMeths)]
#    'numH2Segments':100,

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
    'traitChr':['1'],
    'muEpsRange':[[mu,eps] for mu in muRange for eps in epsRange],
    'numDecScore':3,
    'H1SnpSize':4000,
    'H2SnpSize':300,
    'RPath':'ga/ail/R/',
    'PCIsPreds':False,
    'linBatch':False,
    'cisMean':True,
    'Types':oldMeths+ellMeths,
    'colors':colors,
    'logName':'log/'+opArgs[args[0]]+'-'+str(datetime.now()),
    'RefPerCore':500,
    'RefReps':500000,    
    'numSimStatSegments':10,
    'maxSnpGen':5000,
    'nameForGRM':'comparison',
    'simLearnType':'Full',
    'response':'hipRaw',
    'simGRM':True
}

print(np.array(parmsList)[args],flush=True)
setupFolders(parms,opArgs[args[0]])
name=parms['name']

if parms['makeSimPedFiles']:
    makeSimPedFiles(parms)

if parms['genSimZScores']:
    genSimZScores(parms)
    
if parms['diagnostics']:                 
    plotOffDiag(['Lgrm-all','LsimGrm'],parms)
    plotZ(parms,'simSnps',['1','2'])

if parms['setupELL']:
    setupELL(parms)
        
if parms['genRefStats']:
    genRefStats(parms,ellMeths,'LZCorr')
    genRefStats(parms,oldMeths,'Leye')

if parms['genSimStats']:
    snpChr=[str(2+x) for x in range(len(parms['muEpsRange'])+1)]
    genSimStats(parms,ellMeths,'LZCorr',snpChr)
    genSimStats(parms,oldMeths,'Leye',snpChr)

if parms['makePVals']:
    makePVals(parms)

if parms['plotPower']:
    plotPower(parms)
