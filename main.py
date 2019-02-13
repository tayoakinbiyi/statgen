import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *

warnings.filterwarnings("error")

import numpy as np
import pdb
import os  

def strConcat(df,alpha):
    df=df.merge(alpha,on=['Type'])
    df=df.groupby(['mu','eps','Type']).apply(lambda df:1000*np.nanmean(df.Value>=df.alpha))
    df.name='power'
    df=df.reset_index()

    minDF=df.power.rank(ascending=False,method='min').astype('int').astype(str)
    maxDF=df.power.rank(ascending=False,method='max').astype('int').astype(str)
    concat=pd.Series(minDF.astype('int').astype(str),name='r')
    sel=minDF!=maxDF
    concat[sel]=minDF[sel]+'-'+maxDF[sel]

    df=pd.concat([df,concat],axis=1)    
    df=df.sort_values(by=['mu','eps','r'],ascending=[False,False,True])
    return(df)

def sim(parms,I=False):
    if not I:
        L=np.linalg.cholesky(sig)
    else:
        L=None

    pool = Pool()

    if parms['runAlpha']:
        alpha=pd.DataFrame(monteCarlo(parms['H0'],parms['N'],0,0,sigName,L,parms['Types'],pool)[0])
        alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
        alpha.name='alpha'
        alpha=alpha.reset_index()
        alpha.to_csv('raw-alpha-'+str(parms['N'])+'-'+str(parms['H0'])+'-'+sigName+'.csv',index=False)
    else:
        alpha=pd.read_csv('raw-alpha-'+str(parms['N'])+'-'+str(parms['H0'])+'-'+sigName+'.csv')
        
    if parms['runH01']:
        mc=monteCarlo(parms['H01'],parms['N'],0,0,sigName,L,parms['Types'],pool)
        powerH01=mc[0]
        failH01=mc[1]
        
        powerH01=strConcat(powerH01,alpha)
        powerH01.to_csv('raw-H01-power-'+str(parms['N'])+'-'+str(parms['H01'])+'-'+sigName+'.csv',index=False)

        if len(failH01)>0:        
            failH01=failH01.groupby(['mu','eps','Type']).apply(lambda df:pd.DataFrame({'avgFailRate':1000*np.mean(df.Value),
                'pctAllFail':1000*np.mean(df.Value==1)},index=[0]).astype(int)).reset_index().drop(columns='level_3')    
            failH01.to_csv('raw-H01-fail-'+str(parms['N'])+'-'+str(parms['H01'])+'-'+sigName+'.csv',index=False)
    else:
        powerH01=pd.read_csv('raw-H01-power-'+str(parms['N'])+'-'+str(parms['H01'])+'-'+sigName+'.csv')
        if len(set(powerH01.Type.drop_duplicates())&set(['ggnull','ghc','gbj']))>0:        
            failH01=pd.read_csv('raw-H01-fail-'+str(parms['N'])+'-'+str(parms['H01'])+'-'+sigName+'.csv') 
        else:
            failH01=pd.DataFrame()
        
    if parms['runPower'] in ['append','existing']:
        power=pd.read_csv('raw-power-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'-'+str(parms['muRange'])+'.csv')
        if len(set(power.Type.drop_duplicates())&set(['ggnull','ghc','gbj']))>0:        
            fail=pd.read_csv('raw-fail-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'-'+str(parms['muRange'])+'.csv')
        else:
            fail=pd.DataFrame()
    else:
        power=pd.DataFrame([])
        fail=pd.DataFrame([])
        
    if parms['runPower'] in ['replace','append']:
        t_power=pd.DataFrame()
        t_fail=pd.DataFrame()
        
        for eps in np.unique(parms['epsRange'].round().astype(int)):
            for mu in np.unique(parms['muRange']).round(3):
                mc=monteCarlo(parms['H1'],parms['N'],mu,eps,sigName,L,parms['Types'],pool)
                gc.collect()
                
                t_power=t_power.append(strConcat(mc[0],alpha))
                if len(mc[1])>0:
                    t_fail=t_fail.append(mc[1].groupby(['mu','eps','Type']).apply(lambda df:pd.DataFrame(
                        {'avgFailRate':1000*np.mean(df.Value),'pctAllFail':1000*np.mean(df.Value==1)},
                        index=[0]).astype(int)).reset_index().drop(columns='level_3'))
                    
        pool.close()
        pool.join()            

    if parms['runPower'] =='append':
        power=power.append(t_power)
        fail=fail.append(t_fail)
    elif parms['runPower']=='replace':
        power=t_power
        fail=t_fail
        
    power.to_csv('raw-power-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'-'+str(parms['muRange'])+'.csv',index=False)
    fail.to_csv('raw-fail-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'-'+str(parms['muRange'])+'.csv',index=False)            
        
    power=powerH01.append(power)
    fail=failH01.append(fail)
    
    return(power,fail,parms)

if __name__ == '__main__':
    EXCHANGEABLE=[0]
    NORM_SIG=False
    RAT=False
    MOUSE=False
    parms={
        'Types':None,
        'runAlpha':True,
        'runH01':True,
        'runPower':'append',
        'plot':False,
        'H0':50000,
        'H1':10000,
        'H01':10000,
        'fontsize':17
    }
    '''(400,np.linspace(2,3.5,10),range(2,8)),
            (700,np.linspace(2,3.5,10),range(2,8)),
            (1000,np.linspace(2,3.5,10),range(2,15)),'''
    if len(EXCHANGEABLE)>0:
        for N in [1500]:
            for rho in EXCHANGEABLE:            
                sig,sigName=exchangeable(N,rho)
                if rho==0:
                    Types=['hc','gnull','bj','fdr','minP','score']
                else:
                    Types=['hc','gnull','bj','fdr','minP','score','ggnull','ghc','gbj']
                    
                fileDump(sim({**parms,'sigName':sigName,'Types':Types,'N':N,
                              'muRange':np.linspace(1,3.5,12)[2:4],'epsRange':np.linspace(2,N*.01,12)},True))

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
    

