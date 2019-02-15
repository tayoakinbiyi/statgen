import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

def strConcat(df):
    minDF=df.power.rank(ascending=False,method='min').astype('int').astype(str)
    maxDF=df.power.rank(ascending=False,method='max').astype('int').astype(str)
    concat=pd.Series(minDF.astype('int').astype(str),name='r')
    sel=minDF!=maxDF
    concat[sel]=minDF[sel]+'-'+maxDF[sel]

    df=pd.concat([df,concat],axis=1) 
    return(df)

def sim(parms,I=False):
    if not I:
        L=np.linalg.cholesky(sig)
    else:
        L=None
        
    power=pd.DataFrame()
    fail=pd.DataFrame()

    H0=[(parms['H0'],parms['N'],0,0,sigName,L,parms['Types'],True)]
    H01=[(parms['H01'],parms['N'],0,0,sigName,L,parms['Types'],False)]
    H1=[(parms['H1'],parms['N'],mu,eps,sigName,L,parms['Types'],False) 
        for eps in np.unique(parms['epsRange'].round().astype(int)) for mu in np.unique(parms['muRange']).round(3)]
 
    with ProcessPoolExecutor() as executor:    
        results=executor.map(monteCarlo,H0+H01+H1)  

        for result in results:
            if result['alpha']:
                alpha=result['power'].groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
                alpha.name='alpha'
                alpha=alpha.reset_index()
            else:
                power=power.append(result['power'])
                fail=fail.append(result['fail'])
                            
    power=power.merge(alpha,on='Type')
    power=power.groupby(['mu','eps','Type','alpha'],sort=False).apply(lambda df:1000*np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index().drop(columns='alpha')
    power=power.groupby(['mu','eps'],sort=False).apply(strConcat)
    power=power.sort_values(by=['mu','eps','r'],ascending=[False,False,True])

    if len(fail)>0:
        fail=fail.groupby(['mu','eps','Type']).apply(lambda df:pd.DataFrame(
                {'avgFailRate':1000*np.mean(df.Value),'pctAllFail':1000*np.mean(df.Value==1)},
                index=[0]).astype(int)).reset_index().drop(columns='level_3')

    power.to_csv('raw-power-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'.csv',index=False)
    fail.to_csv('raw-fail-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'.csv',index=False)            
        
    return(power,fail,parms)

if __name__ == '__main__':
    EXCHANGEABLE=[0]
    NORM_SIG=False
    RAT=False
    MOUSE=False
    parms={
        'Types':None,
        'plot':True,
        'H0':10000,
        'H1':10000,
        'H01':10000,
        'fontsize':17
    }

    if len(EXCHANGEABLE)>0:
        for N in [2001]:
            for rho in EXCHANGEABLE:            
                sig,sigName=exchangeable(N,rho)
                if rho==0:
                    Types=['hc','gnull','bj','fdr','minP','score']
                else:
                    Types=['hc','gnull','bj','fdr','minP','score','ggnull','ghc','gbj']
                    
                fileDump(sim({**parms,'sigName':sigName,'Types':Types,'N':N,
                              'muRange':np.linspace(2,3,4),'epsRange':np.linspace(2,N*(.01 if N>1000 else .015),4)},True))

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
        fileDump(sim({**parms,'sigName':sigName,'Types':parms['Types'],'N':N,'muRange':np.linspace(1,3.5,10),
                      'epsRange':np.linspace(2,8,8)},True))

    if RAT:
        N=200
        sig,sigName=raw_data('rat.csv','rat',N)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,Run))
    