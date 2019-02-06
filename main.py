import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *

warnings.filterwarnings("error")

import numpy as np
import pdb

def sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,run=True):
    L=np.linalg.cholesky(sig)
    power=pd.DataFrame([])
    fail=pd.DataFrame([])

    if run:
        mc=monteCarlo(H0,N,0,0,sigName,L)
        power=power.append(mc[0])
        fail=fail.append(mc[1])
        for eps in range(1,int(np.ceil(N*eps_frac))+1):
            for mu in np.linspace(np.sqrt(2*np.log(N))/mu_delta,np.sqrt(2*np.log(N)),mu_delta):
                mc=monteCarlo(H1,N,np.round(mu,3),eps,sigName,L)
                power=power.append(mc[0])
                fail=fail.append(mc[1])

        power.to_csv('raw-power-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv',index=False)
        fail.to_csv('raw-fail-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv',index=False)
    else:
        power=pd.read_csv('raw-power-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv')
        fail=pd.read_csv('raw-fail-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.csv')       
    
    return(power,fail,N,H0,H1,sigName,mu_delta,eps_frac)

if __name__ == '__main__':
    EXCHANGEABLE=[0,.1,.2,.3]
    NORM_SIG=False
    RAT=False
    MOUSE=False
    fontsize=17
    Run=True
    
    eps_frac=.015
    mu_delta=20
    H0=50000
    H1=1000
    
    if len(EXCHANGEABLE)>0:
        N=500       
        for rho in EXCHANGEABLE:
            sig,sigName=exchangeable(N,rho)
            if rho==0:
                Types=['hc','gnull','bj','fdr','minP','score']
            else:
                Types=None
            fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,Run),Types,fontsize)

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
    

