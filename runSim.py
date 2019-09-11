import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.setupFolders import *

from ail.simPython.corrY import *
from ail.simPython.genH1SnpSet import *
from ail.simPython.genH0Y import *
from ail.simPython.genH1ZScores import *
from ail.simPython.genIIDStats import *

from ail.statsPython.setupELL import *

local=os.getcwd()+'/'

ops={
    'corrY':False,
    'genH1SnpSet':False,
    'genH0Y':False,
    'genH0ZScores':False,
    'genH0ZCorr':False,
    'genH1ZScores':False,
    'setupELL':False,
    'genIIDStats':False,
    'genH0Stats':False,
    'genH0Type1':False
}
opsList=list(ops.keys())
opArgs={loc:opslist[loc] for loc in range(len(opsList))}

H1Chr='chr'+str(input('H1 chr number : '))
args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

for arg in args:
    ops[opArgs[arg]]=True

parms={
    'H1Chr':H1Chr,
    'etaGRM':.4,
    'etaError':.6,
    'ellEps':1e-6,
    'local':local,
    'name':'sim-'+H1Chr+'/',
    'cpu':cpu_count(),
    'numPCs':10,
    'smallCpu':3,
    'maxZReps':1000,
    'snpChr':['chr'+str(x) for x in range(1,20) if 'chr'+str(x)!=H1Chr],
    'traitChr':['chr'+str(x) for x in range(1,21) if 'chr'+str(x)!=H1Chr],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo'
}

print(H1Chr,np.array(opArgs)[args],flush=True)
setupFolders(parms)

# 1)
if ops['corrY']:
    corrY(parms)

# 2)
if ops['genH1SnpSet']:
    genH1SnpSet(parms)
    
# 3)
if ops['genH0Y']:
    genH0Y(parms)

if ops['genH0ZScores']:
    score(parms)

if ops['genH0ZCorr']:
    genH0ZCorr(parms)

# 4)
if ops['genH1ZScores']:
    genH1ZScores(parms)
    
#5)
if ops['setupELL']:
    setupELL(parms)
    
if ops['genIIDStats']:
    genIIDStats(parms)

if ops['genH0Stats']:
    genH0Pvals(parms)
    
if ops['genH0Type1']:
    genH0Type1(parms)