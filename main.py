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

    for eps in np.linspace(1,int(np.sqrt(parms['N'])),delta):
        for r in np.linspace(1.0/delta,1,delta):
            power=power.append(monteCarlo({**parms,'mu':np.round(np.sqrt(2*r*np.log(parms['N'])),3),'eps':int(eps)},stats,L))

    power=power.sort_values(by=['eps','mu'],ascending=[False,False])
    power[stats]=power[stats].apply(lambda row:row.rank(ascending=False).astype(str)+'/'+(row*1000).astype(int).astype(str),axis=1)
    return(power)

def failStats(raw,st):
    raw=raw.copy()
    firstCols=raw.columns
    for col in firstCols:
        raw[col+'-AvgPct']=raw[col].mean()
        raw[col+'-PctAll']=(raw[col]==1).mean()
    return(raw)

def convStats(raw,stats,parmKeys):
    alpha=raw.loc[raw['nullHyp'],stats].apply(np.nanpercentile,q=95).to_frame().T
    power=(raw.loc[~raw['nullHyp'],stats]-alpha[stats]>=0).groupby(level=['mu','eps','nullHyp'],sort=False).apply(np.nanmean)
    fail=raw['gbj-fail','ggnull-fail','ghc-fail'].groupby(level=['mu','eps','nullHyp'],sort=False).apply(failStats)
    power=alpha.append(power).merge(fail,left_index=True,right_index=True)
    return(power)

def fileDump(rawStats,parms):
    power=rawStats.groupby(level='name',sort=False).apply(convStats)
    power.to_csv(json.dumps(parms)+'-power.csv')
    rawStats.to_csv(json.dumps(parms)+'-raw.csv')
    
if __name__ == '__main__':
    
    delta=10
    NORM_SIG=True
    RAT=True
    MOUSE=True
               
    if NORM_SIG:
        for N in [1000,2000]:
            parms={'N':N,'H0':5000,'H1':500}

            sig,sigName=np.eye(N),'I'
            fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})

            sig,sigName=norm_sig(N,int(N**1.1))
            fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})

            sig,sigName=norm_sig(N,N**1.2)
            fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})

            sig,sigName=norm_sig(N,int(N**1.3))
            fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})            

    if MOUSE:
        parms={'N':200,'H0':5000,'H1':500}

        sig,sigName=raw_data('mouse.csv','mouse',parms['N'])
        fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})

    if RAT:
        parms={'N':400,'H0':5000,'H1':500}

        sig,sigName=raw_data('rat.csv','rat',parms['N'])
        fileDump(sim(parms,sig,delta),{**parms,'sigName':sigName})
    

