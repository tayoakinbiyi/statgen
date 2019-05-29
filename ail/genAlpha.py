import warnings
import matplotlib
matplotlib.use('agg')

from python.monteCarlo import *
from python.norm_sig import *
from python.makeProb import *
from python.makeL import *

#warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

home='/project/abney/ail/'

files={
    'dataDir':home+'data/',
    'scratchDir':home+'scratch/',
    'gemma':home+'gemma'
}
numPCs=10

os.chdir(home)

#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

if __name__ == '__main__':    
    parms={
        'plot':True,
        'H0':200,
        'fontsize':17,
        'new':False
    }
        
    #parms['sig']=np.readtxt(scratchDir+'corr.txt',sep=',')
    parms['sig']=np.eye(300)
    parms['sigName']='ail'
    parms['N']=len(parms['sig'])

    L=makeL(parms)
    
    ggnullDat,ghcDat=makeProb(L,parms)

    alpha,fail=monteCarlo(L,parms['sigName'],0,0,parms['H0'],ggnullDat,ghcDat)
    alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
    pdb.set_trace()
            