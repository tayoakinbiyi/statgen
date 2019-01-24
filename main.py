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
    for eps in np.linspace(1,int(np.sqrt(N)),delta):
        for r in np.linspace(1.0/delta,1,delta):
            power=power.append(monteCarlo(H1,N,np.round(np.sqrt(2*r*np.log(N)),3),int(eps),sigName,False,L))

    return(power,N,H0,H1,sigName,delta)

if __name__ == '__main__':
    
    NORM_SIG=False
    RAT=True
    MOUSE=True
    
    delta=10
    H0=5000
    H1=500
    
    if NORM_SIG:
        N=1000
        sig,sigName=np.eye(N),'I'
        fileDump(sim(N,H0,H1,sigName,sig,delta))

        sig,sigName=norm_sig(N,1.1)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

        sig,sigName=norm_sig(N,1.2)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

        sig,sigName=norm_sig(N,1.3)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

    if MOUSE:
        N=400
        sig,sigName=raw_data('mouse.csv','mouse',N)
        fileDump(sim(N,H0,H1,sigName,sig,delta))

    if RAT:
        N=200
        sig,sigName=raw_data('rat.csv','rat',N)
        fileDump(sim(N,H0,H1,sigName,sig,delta))
    

