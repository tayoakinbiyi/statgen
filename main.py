import warnings

from sim import *
from fileDump import *
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
        'H0':200,
        'H1':100,
        'H01':100,
        'fontsize':17,
        'new':True
    }
    
    #Types=['hc','gnull','bj','fdr','minP','score']
    #Types=['hc','gnull','bj','fdr','minP','score','ggnull','ghc','gbj']'''

    if len(EXCHANGEABLE)>0:
        for N in [300,1000,1500,2000,2500,3000]:
            for rho in EXCHANGEABLE:            
                sig,_=exchangeable(N,rho)
                sigName='iid-ggnull-ghc'
                    
                fileDump(sim({**parms,'sigName':sigName,'N':N,'sig':sig,'muRange':np.linspace(2,3,10),
                              'epsRange':np.linspace(2,N*(.008 if N>2000 else .01 if N>1000 else .017),10)},True))

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
            'raw-power-([0-9]+)-'+str(H1)+'-'+sigName+'.csv',x)] if y is not None]
        for name,N in fileList:
            tmp=pd.read_csv(name).reset_index(drop=True)
            power=power.append(tmp.merge(pd.DataFrame([N]*len(tmp),index=range(len(tmp)),columns=['N']),left_index=True,
                right_index=True))

        nPlot(power,H1,sigName) 
        