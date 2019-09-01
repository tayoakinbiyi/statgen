import warnings
import matplotlib
matplotlib.use('agg')

from python.monteCarlo import *
from python.norm_sig import *
from python.makeProb import *
from python.makeL import *

from multiprocessing import cpu_count

import numpy as np
import pdb
import os  

def genAlpha(parms,sig):
    sigName=parms['sigName']+'/'
    N=parms['N']
    Rpath=parms['Rpath']
    path=parms['path']
    
    L=makeL(parms,sig)

    print('makeProb')
    ggnullDat,ghcDat=makeProb(L,parms)

    print('monteCarlo')
    alpha,fail=monteCarlo(L,0,0,'H0',ggnullDat,ghcDat,parms)
    alphaPct=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
        
    alpha=pd.DataFrame(alpha.sort_values(by=['Type','Value']).set_index('Type').to_dict())
    ranks=alpha.rank(axis=0,method='average',pct=True,ascending=False)/100
    ranks.columns=[x+'_p' for x in ranks.columns]
    alpha=pd.concat([alpha,ranks],axis=1).iloc[int(len(alpha)*.95):]
    
    alpha.to_csv(path+sigName+'alpha.csv',index=False)
    alphaPct.to_csv(path+sigName+'alphaPct.csv',index=False)
    fail.to_csv(path+sigName+'fail.csv',index=False)
    
    return(alpha,alphaPct,fail)
       