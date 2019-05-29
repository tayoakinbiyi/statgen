import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

from python.sim import *
from python.fileDump import *
from python.raw_data import *
import re
import time
from statsmodels.stats.moment_helpers import cov2corr

warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

home='/project/abney/'

files={
    'dataDir':home+'ail/data/',
    'scratchDir':home+'ail/scratch/',
    'gemma':home+'ail/gemma'
}
numPCs=10


#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

if __name__ == '__main__':    
    chrs=['chr'+str(x) for x in range(1,20)]

    pdb.set_trace()
    pvals=pd.DataFrame()
    for traitChr in chrs[0:3]:
        pvalsCol=[]
        for snpChr in chrs[0:3]:
            if traitChr==snpChr:
                continue
            pvalsCol+=[pd.read_csv(scratchDir+'pvals-final-'+snpChr+'-'+traitChr+'.txt',index_col=[0,1],header=[0,1,2])]
     
        pvals=pd.concat([pvals,pd.concat(pvalsCol,axis=0)],axis=1)  
    
    corr=np.ma.corrcoef(pvals,rowvar=False)
    corr=(corr+corr.T)/2
    
    U,D,Vt=np.linalg.svd(corr)
    
    D+=min(D)
    
    corr=np.matmul(np.matmul(U,np.diag(D)),U.T)
    
    corr=cov2corr(corr)
    
    np.savetxt(scratchDir+'corr.txt',sep=',')
    
            