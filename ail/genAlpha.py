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

if __name__ == '__main__':    
    parms={
        'plot':True,
        'H0':200,
        'fontsize':17,
        'new':True
    }
        
    #parms['sig']=np.readtxt(scratchDir+'corr.txt',sep=',')
    sig=np.eye(300)
    parms['sigName']='ail1'
    parms['N']=len(sig)
    parms['path']='/project/abney/ail/'
    parms['Rpath']='/home/akinbiyi/ail/ail/R'
    parms['cpus']=2

    L=makeL(parms,sig)
    
    ggnullDat,ghcDat=makeProb(L,parms)

    alpha,fail=monteCarlo(L,0,0,'H0',ggnullDat,ghcDat,parms)
    alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
    pdb.set_trace()
            