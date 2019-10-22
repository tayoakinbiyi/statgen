import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.callFuncs import *
    
local=os.getcwd()+'/'

ops={
    'process':False,
    'score':False,
    'corr':False,
    'qq':False,
    'z2':False,
    'man':False,
    'usThem':False,
}
opsList=list(ops.keys())
opArgs={loc:opsList[loc] for loc in range(len(opsList))}

parmsAll={}

parmsAll['comparison/']={
    'response':'hipExp',
    'quantNormalizeExpr':False,
    'remPCFromSnp':False,
    'remPCFromTraits':False,
    'remPCCorrSnp':False,
    'PCIsPreds':False,
    'CovIsPreds':True,
    'remCovFromTraits':False,
    'grmParm':'c',
    'wald':False,
    'linBatch':True,
    'traitChr':['chr1']
}

parmsAll['remPredPCInGemma/']={
    'response':'hipRaw',
    'quantNormalizeExpr':True,
    'remPCFromSnp':False,
    'remPCFromTraits':False,
    'remPCCorrSnp':False,
    'PCIsPreds':True,
    'CovIsPreds':False,
    'remCovFromTraits':True,
    'grmParm':'s',
    'linBatch':False,
    'wald':True
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

parms={
    **ops,
    'local':local,
    'name':name,
    'cpu':cpu_count(),
    'numPCs':10,
    'smallCpu':3,
    'snpChr':['chr'+str(x) for x in range(1,20)],
    'traitChr':['chr'+str(x) for x in range(1,21)],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo',
    'allChrGRM':False,
    'cisMean':False,
    **parmsAll[name]
}

callFuncs(parms)

