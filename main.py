import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *

#warnings.filterwarnings("error")

import numpy as np
import scipy
import plotly.plotly as py
import plotly.graph_objs as go

def sim(N,H0,H1,sigName,sig,delta):
    L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)
    power=pd.DataFrame([])

    power=power.append(monteCarlo(H0,N,0,0,sigName,True,L))
    for eps in np.linspace(1,int(np.sqrt(N)),2):
        for r in np.linspace(1.0/delta,1,2):
            power=power.append(monteCarlo(H1,N,np.round(np.sqrt(2*r*np.log(N)),3),int(eps),sigName,False,L))

    return(power,N,H0,H1,sigName)

if __name__ == '__main__':
    
    delta=10
    NORM_SIG=True
    RAT=False
    MOUSE=False
    
    H0=100
    H1=100
    
    #pdb.set_trace()
    #N=1000;sig,sigName=norm_sig(N,1.1);L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)   
    
    if NORM_SIG:
        for N in [40]:
            sig,sigName=np.eye(N),'I'
            fileDump(sim(N,H0,H1,sigName,sig,delta))

            sig,sigName=norm_sig(N,1.1)
            fileDump(sim(N,H0,H1,sigName,sig,delta))

            sig,sigName=norm_sig(N,1.2)
            fileDump(sim(N,H0,H1,sigName,sig,delta))

            sig,sigName=norm_sig(N,1.3)
            fileDump(sim(N,H0,H1,sigName,sig,delta))

    if MOUSE:
        parms={'N':200,'H0':5000,'H1':500}

        sig,sigName=raw_data('mouse.csv','mouse',parms['N'])
        fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})

    if RAT:
        parms={'N':400,'H0':5000,'H1':500}

        sig,sigName=raw_data('rat.csv','rat',parms['N'])
        fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})
    

