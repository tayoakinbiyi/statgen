import warnings
import numpy as np
from genL import *
from monteCarlo import *
#snap-05e3c2f3c7efd6df1
from mymath import *
#warnings.filterwarnings("error")
import scipy.stats as st
from norm_sig import *

def sim(N,delta,H0,H1,sigName,sig,stats):
    L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)
    power=pd.DataFrame([])

    for theta in np.linspace(.5+0.5/delta,1-0.5/delta,delta):
        for r in np.linspace(rho(theta)+(theta-rho(theta))/delta,theta-(theta-rho(theta))/delta,delta):
            parms={'H0':H0,'H1':H1,'N':N,'mu':np.round(np.sqrt(2*r*np.log(N)),3),'eps':int(N**(1-theta)),
                 'r':np.round(r,3),'theta':np.round(theta,3),'sig':sigName}
            power=power.append(monteCarlo(parms,stats,L))

    power=power.sort_values(by=['eps','mu'],ascending=[False,False])
    power[stats]=power[stats].apply(lambda row:row.rank(ascending=False).astype(str)+'/'+(row*1000).astype(int).astype(str),axis=1)
    return(power)
    
if __name__ == '__main__':
    
    stats=['ghc','hc','bj','gbj','gnull','ggnull','cpma','score','alr','fdr_ratio','minP']
    N=100
    delta=2
    H0=200
    H1=200
    power=pd.DataFrame()
    
    #sigName,sig='I',np.eye(N)  
    #power=power.append(sim(N,delta,H0,H1,sigName,sig,stats))
    #sig=np.loadtxt(sigName+'.csv', delimiter=",")

    sigName,sig='norm_sig_1',norm_sig(N,.3)
    power=power.append(sim(N,delta,H0,H1,sigName,sig,stats))

    sigName,sig='norm_sig_2',norm_sig(N,20)
    power=power.append(sim(N,delta,H0,H1,sigName,sig,stats))

    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)
    
