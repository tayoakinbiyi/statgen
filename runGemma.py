import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import pdb
from multiprocessing import cpu_count

from ail.opPython.callFuncs import *

import json
    
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
opArgs=list(ops.keys())

parmsAll={}

args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

for arg in args:
    ops[opArgs[arg]]=True

parmsAll['natalia/']={
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
    'subsetFirstGRM':True,
    'traitChr':['chr'+str(x) for x in range(1,2)]
}

parmsAll['nataliaFullGRM/']={
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
    'subsetFirstGRM':False,
    'traitChr':['chr'+str(x) for x in range(1,2)]
}

parmsAll['remPCExprAndSnp/']={
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'remPCFromSnp':True,
    'remPCFromTraits':True,
    'remPCCorrSnp':True,
    'PCIsPreds':False,
    'CovIsPreds':False,
    'remCovFromTraits':True,
    'grmParm':'s',
    'wald':True,
    'subsetFirstGRM':False
}

parmsAll['remPCJustTrait/']={
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'remPCFromSnp':False,
    'remPCFromTraits':True,
    'remPCCorrSnp':True,
    'PCIsPreds':False,
    'CovIsPreds':False,
    'remCovFromTraits':True,
    'grmParm':'s',
    'wald':True,
    'subsetFirstGRM':False
}

#########################################################################################33

names=list(parmsAll.keys())
name=names[int(input(str(names)+' : '))]

print(name,np.array(opArgs)[args],flush=True)

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
    'allChrGRM':False
    **parmsAll[name]
}

with open(local+'op','w') as f:
    f.write(json.dumps(parms))

callFuncs(parms)

