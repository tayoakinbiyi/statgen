import warnings
import matplotlib
matplotlib.use('agg')

from python.monteCarlo import *
from python.norm_sig import *
from python.genAlpha import *

import numpy as np
import pdb
import os  

home='/home/akinbiyi/ail/'
#home='/home/ubuntu/ail/'

files={
    'dataDir':home+'data/',
    'scratchDir':home+'scratch/',
    'gemma':home+'gemma'
}

#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']
traitInfo=pd.read_csv(scratchDir+'traitInfo.csv')    

sig=np.eye(300)
parms={
    'plot':True,
    'H0':20,
    'fontsize':17,
    'new':True,
    'path':home,
    'Rpath':home+'ggnull/R',
    'cpus':cpu_count(),
    'binPower':500,
    'eps':1e-10,
    'delta':.1,
    'H0':30,
    'dataDir':files['dataDir'],
    'scratchDir':files['scratchDir'],
    'sigName':'test',
    'N':len(sig)
}

'''
for trait in ['chr'+x for x in range(1,20)]:
    parms['sigName']='ailfit-'+trait

    sig=np.loadtxt(scratchDir+'corr.csv',delimiter=',')    
    sig=sig[traitInfo.chromosome!=parms['chr'],traitInfo.chromosome!=parms['chr']]
    
    alpha,alphaPct,fail=genAlpha(parms,sig)
    parms['N']=len(sig)
'''    
alpha,alphaPct,fail=genAlpha(parms,sig)            