import warnings
import numpy as np
from genL import *
from monteCarlo import *
#snap-05e3c2f3c7efd6df1
from mymath import *
#warnings.filterwarnings("error")
import scipy.stats as st
from norm_sig import *

if __name__ == '__main__':
    
    N=30
    delta=10
    H0=200
    H1=100
    
    sigName='norm_sig'
    #sig=np.eye(N)    
    #sig=np.loadtxt(sigName+'.csv', delimiter=",")
    sig=norm_sig(N)
    L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)

    power=pd.DataFrame([])

    for theta in np.linspace(.5+0.5/delta,1-0.5/delta,delta)[0:2]:
        for r in np.linspace(rho(theta)+(theta-rho(theta))/delta,theta-(theta-rho(theta))/delta,delta)[0:2]:
            parms={'H0':H0,'H1':H1,'N':N,'mu':np.round(np.sqrt(2*r*np.log(N)),3),'eps':int(N**(1-theta)),
                 'r':np.round(r,3),'theta':np.round(theta,3),'sig':sigName}
            power=power.append(monteCarlo(parms,L))

    power.to_csv('power.csv',index=False)