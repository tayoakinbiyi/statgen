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

parmsAll={}

parmsAll['Wald/']={
    'pval':'wald',
    'name':'Wald/'
}

parmsAll['LRT/']={
    'pval':'lrt',
    'name':'LRT/'
}

parmsAll['Z/']={
    'pval':'z',
    'name':'Z/'
}

baseParms={
    'response':'hipExp',
    'quantNormalizeExpr':False,
    'remPCFromSnp':False,
    'remPCFromTraits':False,
    'remPCCorrSnp':False,
    'PCIsPreds':False,
    'CovIsPreds':True,
    'remCovFromTraits':False,
    'grmParm':'c',
    'linBatch':False,
    'local':local,
    'cpu':cpu_count(),
    'numPCs':10,
    'smallCpu':3,
    'snpChr':['chr1'],
    'traitChr':['chr1'],
    'snpFile':'ail.genos.dosage.gwasSNPs.txt',
    'numDecScore':3,
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo',
    'allChrGRM':False,
    'cisMean':False,
}

fig,axs=plt.subplots(1,1,dpi=50,tight_layout=True)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
'''
for key in parmsAll:
    parms={**baseParms,**parmsAll[key]}
    #setupFolders(parms)
    #process(parms)
    #score(parms)
    if key=='Z/':    
        p+=[2*norm.sf(np.abs(DBRead(key+'score/p-chr1-chr1',parms,True).flatten()))
    else:
        p=DBRead(key+'score/p-chr1-chr1',parms,True).flatten()
'''    
#x=DBRead('Wald/score/p-chr1-chr1',{**baseParms,**parmsAll['Wald/']},True)
#pdb.set_trace()
pZ=-np.log10(2*norm.sf(np.abs(DBRead('Z/score/p-chr1-chr1',{**baseParms,**parmsAll['Z/']},True)[:,0:100].flatten())))
#pWald=-np.log10(DBRead('Wald/score/p-chr1-chr1',{**baseParms,**parmsAll['Wald/']},True)[:,0:100].flatten())
pLRT=-np.log10(DBRead('LRT/score/p-chr1-chr1',{**baseParms,**parmsAll['LRT/']},True)[:,0:100].flatten())

mMax=max(max(pZ),max(pLRT))
mMin=min(min(pZ),min(pLRT))

axs.scatter(pLRT,pZ,label='Z~LRT')
#axs.scatter(pLRT,pWald,label='Wald~LRT')
axs.plot([mMin,mMax],[mMin,mMax],ls="--", c=".3")    
    
axs.legend()
fig.savefig('testWaldLRT.png')
plt.close('all')
