import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *

#warnings.filterwarnings("error")

import numpy as np
import pdb

def sim(N,H0,H1,sigName,sig,mu_delta,eps_delta):
    L=np.linalg.cholesky(sig)
    power=pd.DataFrame([])

    power=power.append(monteCarlo(H0,N,0,0,sigName,L))
    for eps in np.linspace(1,np.ceil(N*.01),eps_delta).round().astype(int):
        for mu in np.linspace(np.sqrt(2*np.log(N))/mu_delta,np.sqrt(2*np.log(N)),mu_delta):
            power=power.append(monteCarlo(H1,N,np.round(mu,3),eps,sigName,L))
    
    return(power,N,H0,H1,sigName)

if __name__ == '__main__':
    I=True
    EXCHANGEABLE=True
    NORM_SIG=False
    RAT=True
    MOUSE=True
    
    eps_delta=4
    mu_delta=10
    H0=50000
    H1=1000
    
    if I:
        N=400
        sig,sigName=np.eye(N),'I'
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_delta),Types=np.array(['bj','fdr_ratio','hc','score','minP','gnull']))
        
    if EXCHANGEABLE:
        N=400
        sig,sigName=exchangeable(N,.1)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_delta))
        
        sig,sigName=exchangeable(N,.2)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_delta))

        sig,sigName=exchangeable(N,.4)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_delta))

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
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_delta))

    if RAT:
        N=200
        sig,sigName=raw_data('rat.csv','rat',N)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_delta))
    

