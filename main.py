import warnings
import numpy as np
from genL import *
from monteCarlo import *
from mymath import *
#warnings.filterwarnings("error")
import scipy.stats as st
from norm_sig import *

def sim(parms,sig,delta):
    stats=['ghc','hc','bj','gbj','gnull','ggnull','score','fdr_ratio','minP']
    L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)
    power=pd.DataFrame([])
    eps=0

    for theta in np.linspace(.5,1,delta):
        t_eps=int(N**(1-theta))
        if t_eps==eps:
            continue
        eps=t_eps
        for r in np.linspace(1.0/delta,1,delta):
            power=power.append(monteCarlo({**parms,'mu':np.round(np.sqrt(2*r*np.log(parms['N'])),3),'eps':eps,'asymp':'undetectable' if 
                r<=rho(theta) else 'detectable' if rho(theta)<r<theta else 'identifiable'},stats,L))

    power=power.sort_values(by=['eps','mu'],ascending=[False,False])
    power[stats]=power[stats].apply(lambda row:row.rank(ascending=False).astype(str)+'/'+(row*1000).astype(int).astype(str),axis=1)
    return(power)
    
if __name__ == '__main__':
    
    N=400
    delta=10
    H0=5000
    H1=500
    
    parms={'N':N,'H0':H0,'H1':H1}
    
    power=pd.DataFrame()
    
    sig,sigParms=np.eye(N),{'name':'I','min_cor':0,'avg_cor':0,'max_cor':0}
    power=power.append(sim({**parms,**sigParms},sig,delta))
    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)

    sig,sigParms=norm_sig(N,int(N**1.1))
    power=power.append(sim({**parms,**sigParms},sig,delta))
    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)

    sig,sigParms=norm_sig(N,N**1.2)
    power=power.append(sim({**parms,**sigParms},sig,delta))
    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)

    sig,sigParms=norm_sig(N,int(N**1.3))
    power=power.append(sim({**parms,**sigParms},sig,delta))
    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)

    sig,sigParms=raw_data('mouse.csv','mouse',parms)
    power=power.append(sim({**parms,**sigParms},sig,delta))
    power.to_csv(str(parms['N'])+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)

    parms['N']=200
    sig,sigParms=raw_data('rat.csv','rat',parms)
    power=power.append(sim({**parms,**sigParms},sig,delta))
    power.to_csv(str(parms['N'])+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)
