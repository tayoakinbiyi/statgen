import warnings
import numpy as np
from genL import *
from monteCarlo import *
#snap-05e3c2f3c7efd6df1
from mymath import *
warnings.filterwarnings("error")
import scipy.stats as st
from norm_sig import *

def sim(parms,sigParms,sig,stats):
    stats=['ghc','hc','bj','gbj','gnull','ggnull','cpma','score','alr','fdr_ratio','minP']
    L=scipy.linalg.cholesky(sig,overwrite_a=True, check_finite=False)
    power=pd.DataFrame([])

    for theta in np.linspace(.5+0.5/delta,1-0.5/delta,delta):
        for r in np.linspace(rho(theta)+(theta-rho(theta))/delta,theta-(theta-rho(theta))/delta,delta):
            parms={**parms,**sigParts,
                   'mu':np.round(np.sqrt(2*r*np.log(N)),3),'eps':int(N**(1-theta))}
            power=power.append(monteCarlo(parms,stats,L))

    power=power.sort_values(by=['eps','mu'],ascending=[False,False])
    power[stats]=power[stats].apply(lambda row:row.rank(ascending=False).astype(str)+'/'+(row*1000).astype(int).astype(str),axis=1)
    return(power)
    
if __name__ == '__main__':
    
    N=100
    delta=2
    H0=100
    H1=100
    
    parms={'N':N,'delta':delta,'H0':H0,'H1':H1}
    
    power=pd.DataFrame()
    
    sig,sigParms=np.eye(N),{'pct_neg_cor':0,'min_cor':0,'avg_cor':0,'max_cor':0}
    power=power.append(sim(parms,sigParms,sig,stats))

    sig,sigParms=norm_sig(N,0)
    power=power.append(sim({**parms,sigParms},sig,stats))
    
    sig,sigParms=norm_sig(N,N)
    power=power.append(sim({**parms,sigParms},sig,stats))
    
    sig,sigParms=norm_sig(N,int(N**1.5))
    power=power.append(sim({**parms,sigParms},sig,stats))

    sig,sigParms=norm_sig(N,N**2)
    power=power.append(sim({**parms,sigParms},sig,stats))

    sig,sigParms=norm_sig(N,int(N**2.5))
    power=power.append(sim({**parms,sigParms},sig,stats))

    sig,sigParms=rat_data(N)

    power.to_csv(str(N)+'-'+str(H0)+'-'+str(H1)+'.csv',index=False)
    
