import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

from python.sim import *
from python.fileDump import *
import re
import time

warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

if __name__ == '__main__':
    EXCHANGEABLE=[0]
    NORM_SIG=False
    RAT=False
    MOUSE=False
    CROSS_N='iid-ggnull-ghc'
    
    parms={
        'Types':['hc','gnull','bj','fdr','minP','score','ggnull','ghc'],
        'plot':True,
        'H0':50000,
        'H01':10000,
        'H1':10000,
        'fontsize':17,
        'new':True
    }
    
    if len(EXCHANGEABLE)>0:
        for N in [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,10000]:
            for rho in EXCHANGEABLE:            
                sig,_=exchangeable(N,rho)
                sigName='iid-ggnull-ghc-'+str(N)
                
                #np.unique(np.linspace(2,3,10)).round(3)
                #np.unique(np.linspace(2,N*(.008 if N>2000 else .01 if N>1000 else .017),10).round()).astype(int)
                fileDump(sim({**parms,'sigName':sigName,'N':N,'sig':sig,'muRange':np.array([1.8,2.04,2.28,2.52,2.76,3]),
                    'epsRange':np.array([2,5,9,12,16,20]).astype(int)}))

    if NORM_SIG:
        N=1000
        sig,sigName=norm_sig(N,1.1)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

        sig,sigName=norm_sig(N,1.2)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

        sig,sigName=norm_sig(N,1.3)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

    if MOUSE:
        N=300
        sig,sigName=raw_data('mouse.csv','mouse',N)
        fileDump(sim({**parms,'sigName':sigName,'N':N,'sig':sig,'muRange':np.linspace(1,3.5,10),
                      'epsRange':np.linspace(2,8,8)},True))

    if RAT:
        N=200
        sig,sigName=raw_data('rat.csv','rat',N)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,Run))
    
    if CROSS_N is not None:
        sigName=CROSS_N
        H1=parms['H1']
        
        power=pd.DataFrame()
        
        fileList=[(y.group(0),int(y.group(1))) for x in os.listdir() for y in [re.search(
            '../raw/raw-power-'+sigName+'-([0-9]+).csv',x)] if y is not None]
        for name,N in fileList:
            tmp=pd.read_csv(name).reset_index(drop=True)
            power=power.append(tmp.merge(pd.DataFrame([N]*len(tmp),index=range(len(tmp)),columns=['N']),left_index=True,
                right_index=True))

        nPlot(power,H1,sigName) 
        