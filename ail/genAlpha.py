import warnings
import matplotlib
matplotlib.use('agg')

from python.monteCarlo import *
from python.norm_sig import *
from python.makeProb import *
from python.makeL import *

from multiprocessing import cpu_count

#warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

def genAlpha():
    parms={
        'plot':True,
        'H0':20,
        'fontsize':17,
        'new':True
    }
    
    home='/home/ubuntu/ail/'
    files={
        'dataDir':home+'data/',
        'scratchDir':home+'scratch/',
        'gemma':home+'gemma'
    }

    #load raw files
    dataDir=files['dataDir']
    scratchDir=files['scratchDir']
        
    sig=np.loadtxt(scratchDir+'corr.csv',delimiter=',')
    parms['sig']=sig
    #sig=np.eye(300)
    parms['sigName']='ailfit'
    parms['N']=len(sig)
    parms['path']=home
    parms['Rpath']=home+'ail/R'
    parms['cpus']=cpu_count()

    print('L')
    L=makeL(parms,sig)
    
    print('makeProb')
    ggnullDat,ghcDat=makeProb(L,parms)

    print('monteCarlo')
    alpha,fail=monteCarlo(L,0,0,'H0',ggnullDat,ghcDat,parms)
    alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95)).to_csv(sigName+'/alpha_pct.csv')
    alpha.to_csv(sigName+'/alpha.csv')
    
if __name__ == '__main__':    
    genAlpha()

            