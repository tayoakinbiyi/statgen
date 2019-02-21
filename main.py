import warnings

from monteCarlo import *
from fileDump import *
from norm_sig import *
import re

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

def powerMat(dat,alpha):
    power,fail=dat
    power=power.merge(alpha,on='Type')
    power=power.groupby(['mu','eps','Type','alpha'],sort=False).apply(lambda df:1000*np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index().drop(columns='alpha')
    power=strConcat(power)
    power=power.sort_values(by=['mu','eps','r'],ascending=[False,False,True])

    if len(fail)>0:
        fail=fail.groupby(['mu','eps','Type'],sort=False).apply(lambda df:pd.DataFrame({'avgFailRate':1000*np.mean(df.Value),
            'pctAllFail':1000*np.mean(df.Value==1)},index=[0]).astype(int)).reset_index()
        if sum(fail.columns.str.contains('level'))>0:
            fail=fail.drop(columns=fail.columns[fail.columns.str.contains('level')])
        
    return power,fail

def sim(parms,I=False):
    L=np.linalg.cholesky(parms['sig'])
    z=np.matmul(L.T,np.random.normal(0,1,size=(N,max(parms['H0'],parms['H1'],parms['H01'])))).T
    empSig=np.corrcoef(z,rowvar=False)
    sig_tri=empSig[np.triu_indices(N,1)].flatten()
        
    if parms['new']:
        alpha,_=monteCarlo(parms['H0'],{**parms,'sig_tri':sig_tri},0,0,z[0:parms['H0'],:])
        alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
        alpha.name='alpha'
        alpha=alpha.reset_index()

        power, fail=powerMat(monteCarlo(parms['H01'],{**parms,'sig_tri':sig_tri},0,0,z[0:parms['H01'],:]),alpha)

        epsMu=[(eps,mu) for eps in np.unique(parms['epsRange'].round().astype(int)) for mu in np.unique(parms['muRange']).round(3)]
        epsMu=sorted(epsMu,key=lambda x: (x[0]-len(np.unique(parms['epsRange']))/2)**2+(x[1]-len(np.unique(parms['muRange']))/2)**2)
        
        while len(epsMu)>0:
            eps,mu=epsMu[0]
            mc=powerMat(monteCarlo(parms['H1'],{**parms,'sig_tri':sig_tri},mu,eps,z[0:parms['H1'],:]),alpha)
            power=power.append(mc[0])
            fail=fail.append(mc[1])

            epsMu=epsMu[1:]         
            if mc[0].power.max()>850:
                power=power.append(pd.DataFrame([[t_mu,t_eps,Type,1000,'1-'+str(len(parms['Types']))] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps>eps and t_mu>mu)],columns=power.columns))
                epsMu=[(t_eps,t_mu) for (t_eps,t_mu) in epsMu if not (t_eps>eps and t_mu>mu)]
            if mc[0].power.max()<350:
                power=power.append(pd.DataFrame([[t_mu,t_eps,Type,1,'1-'+str(len(parms['Types']))] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps<eps and t_mu<mu)],columns=power.columns))
                epsMu=[(t_eps,t_mu) for (t_eps,t_mu) in epsMu if not (t_eps<eps and t_mu<mu)]
                            
        power.to_csv('raw-power-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'.csv',index=False)
        fail.to_csv('raw-fail-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'.csv',index=False)            
    else:
        power=pd.read_csv('raw-power-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'.csv')
        fail=pd.read_csv('raw-fail-'+str(parms['N'])+'-'+str(parms['H1'])+'-'+sigName+'.csv')            
        
    return(power,fail,parms)

if __name__ == '__main__':
    EXCHANGEABLE=[0]
    NORM_SIG=False
    RAT=False
    MOUSE=False
    CROSS_N='iid-ggnull-ghc'
    
    parms={
        'Types':['hc','gnull','bj','fdr','minP','score','ggnull','ghc'],
        'plot':True,
        'H0':50,
        'H1':10,
        'H01':10,
        'fontsize':17,
        'new':True
    }
    
    #Types=['hc','gnull','bj','fdr','minP','score']
    #Types=['hc','gnull','bj','fdr','minP','score','ggnull','ghc','gbj']'''

    if len(EXCHANGEABLE)>0:
        for N in [100,1000,1500,2000,2500,3000]:
            for rho in EXCHANGEABLE:            
                sig,_=exchangeable(N,rho)
                sigName='iid-ggnull-ghc'
                    
                fileDump(sim({**parms,'sigName':sigName,'N':N,'sig':sig,'muRange':np.linspace(2,3,10),
                              'epsRange':np.linspace(2,N*(.008 if N>2000 else .01 if N>1000 else .017),10)},True))

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
        fileDump(sim({**parms,'sigName':sigName,'N':N,'sig':sig,'muRange':np.linspace(1,3.5,10),
                      'epsRange':np.linspace(2,8,8)},True))

    if RAT:
        N=200
        sig,sigName=raw_data('rat.csv','rat',N)
        fileDump(sim(N,H0,H1,sigName,sig,mu_delta,eps_frac,Run))
    
    if CROSS_N is not None:
        sigName=CROSS_N
        H1=parms['H1']
        
        power=pd.DataFrame()
        
        fileList=[(y.group(0),int(y.group(1))) for x in os.listdir() for y in [re.search(
            'raw-power-([0-9]+)-'+str(H1)+'-'+sigName+'.csv',x)] if y is not None]
        for name,N in fileList:
            tmp=pd.read_csv(name).reset_index(drop=True)
            power=power.append(tmp.merge(pd.DataFrame([N]*len(tmp),index=range(len(tmp)),columns=['N']),left_index=True,
                right_index=True))

        nPlot(power,H1,sigName) 
        