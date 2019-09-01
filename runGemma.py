import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import pdb
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from multiprocessing import cpu_count
import warnings
#warnings.simplefilter("error")
import argparse
import dropbox

from ail.callFuncs import *
from ail.opPython.DB import *

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

names=['natalia/','remPCExprAndSnp/','remPCJustTrait/']
parmsAll={}

name=names[int(input(str(names)+' : '))]
args=[int(x) for x in input(str(opArgs)+': ').split(' ')]

print(name,np.array(opArgs)[args],flush=True)
if not os.path.exists(local+name):
    os.mkdir(local+name)

dbx = dropbox.Dropbox('YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo')
dbx.users_get_current_account()

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
    'wald':False
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
    'wald':False
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
    'wald':False
}

#########################################################################################33

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
    **parmsAll[name]
}

with open(local+'op','w') as f:
    f.write(json.dumps(parms))

callFuncs({**parms,'dbx':dbx})

