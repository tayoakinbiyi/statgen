import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *

warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

def sim(N,H0,H1,H01,sigName,sig,mu_delta,eps_frac,run=True):
    L=np.linalg.cholesky(sig)
    power=pd.DataFrame([])
    fail=pd.DataFrame([])

    if os.path.isfile('raw-alpha-'+str(N)+'-'+str(H0)+'-'+sigName+'.csv'):
        alpha=pd.read_csv('raw-alpha-'+str(N)+'-'+str(H0)+'-'+sigName+'.csv')
    else:
        alpha=pd.DataFrame(monteCarlo(H0,N,0,0,sigName,L)[0])
        alpha.to_csv('raw-alpha-'+str(N)+'-'+str(H0)+'-'+sigName+'.csv',index=False)
        
    if run:
        mc=monteCarlo(H01,N,0,0,sigName,L)
        power=power.append(mc[0])
        fail=fail.append(mc[1])
        for eps in range(2,int(np.ceil(N*eps_frac))+1):
            for mu in np.linspace(1,3,mu_delta):
                mc=monteCarlo(H1,N,np.round(mu,3),eps,sigName,L)
                power=power.append(mc[0])
                fail=fail.append(mc[1])

        power.to_csv('raw-power-'+str(N)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv',index=False)
        fail.to_csv('raw-fail-'+str(N)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv',index=False)
    else:
        power=pd.read_csv('raw-power-'+str(N)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv')
        fail=pd.read_csv('raw-fail-'+str(N)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv')       
    
    return(alpha,power,fail,N,H0,H1,H01,sigName,mu_delta,eps_frac)

if __name__ == '__main__':
    EXCHANGEABLE=[0]
    NORM_SIG=False
    RAT=False
    MOUSE=False
    fontsize=17
    Run=False
    
    eps_frac=.01
    mu_delta=8
    H0=50000
    H1=1000
    H01=10000
    
    if len(EXCHANGEABLE)>0:
        N=700       
        for rho in EXCHANGEABLE:
            sig,sigName=exchangeable(N,rho)
            if rho==0:
                Types=['hc','gnull','bj','fdr','minP','score']
            else:
                Types=None
            fileDump(sim(N,H0,H1,H01,sigName,sig,mu_delta,eps_frac,Run),Types,fontsize)

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
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,Run))

    if RAT:
        N=200
        sig,sigName=raw_data('rat.csv','rat',N)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,Run))
    

