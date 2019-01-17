import warnings
import numpy as np
from genL import *
from monteCarlo import *
from mymath import *
#warnings.filterwarnings("error")
import scipy.stats as st
from norm_sig import *

def sim(parms,sigParms,sig,delta):
    stats=['ghc','hc','bj','gbj','gnull','ggnull','cpma','score','fdr_ratio','minP']
    L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)
    power=pd.DataFrame([])
    parms.update(sigParms)

    for theta in np.linspace(.5,1,delta):
        for r in np.linspace(1.0/delta,1,delta):
            t_parms=parms
            t_parms.update({'mu':np.round(np.sqrt(2*r*np.log(N)),3),'eps':int(N**(1-theta)),'asymp':'unidentifiable' if r<=rho(theta) 
                       else 'identifiable' if rho(theta)<r<theta else 'detectable'})
            power=power.append(monteCarlo(t_parms,stats,L))

    power=power.sort_values(by=['eps','mu'],ascending=[False,False])
    power[stats]=power[stats].apply(lambda row:row.rank(ascending=False).astype(str)+'/'+(row*1000).astype(int).astype(str),axis=1)
    return(power)
    
if __name__ == '__main__':
    
    N=40
    delta=3
    H0=100
    H1=100
    
    parms={'N':N,'H0':H0,'H1':H1}
    
    power=pd.DataFrame()
    
    sig,sigParms=np.eye(N),{'name':'I','min_cor':0,'avg_cor':0,'max_cor':0}
    parms.update(sigParms)
    power=power.append(sim(parms,sigParms,sig,delta))

    sig,sigParms=norm_sig(N,int(N**1.5))
    power=power.append(sim(parms,sigParms,sig,delta))

    sig,sigParms=norm_sig(N,N**1.75)
    power=power.append(sim(parms,sigParms,sig,delta))

    sig,sigParms=norm_sig(N,int(N**2))
    power=power.append(sim(parms,sigParms,sig,delta))

    sig,sigParms=rat_data(N)
    power=power.append(sim(parms,sigParms,sig,delta))

    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)
    
