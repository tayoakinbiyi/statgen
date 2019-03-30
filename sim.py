from monteCarlo import *
from norm_sig import *
from makeProb import *
from functools import partial

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
    power=power.groupby(['eps','mu','Type','alpha'],sort=False).apply(lambda df:1000*np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index().drop(labels='alpha',axis=1)
    power=strConcat(power)
    power=power.sort_values(by=['eps','mu','r'],ascending=[False,False,True])

    if len(fail)>0:
        fail=fail.groupby(['eps','mu','Type'],sort=False).apply(lambda df:pd.DataFrame({'avgFailRate':1000*np.mean(df.Value),
            'pctAllFail':1000*np.mean(df.Value==1)},index=[0]).astype(int)).reset_index()
        if sum(fail.columns.str.contains('level'))>0:
            fail=fail.drop(labels=fail.columns[fail.columns.str.contains('level')],axis=1)
        
    return power,fail

def sim(parms):
    N=parms['N']
    L=np.linalg.cholesky(parms['sig'])  
    
    epsRange=parms['epsRange']
    muRange=parms['muRange']
    sigName=parms['sigName']
    
    ebb,var=makeProb(L,parms)
    
    if parms['new']:
        alpha,_=monteCarlo(L,sigName,0,0,parms['H0'],ebb,var)
        
        print('alpha', psutil.virtual_memory().percent)
        ggnull=alpha[alpha.Type=='ggnull']
        ggnull0=alpha[alpha.Type=='ggnull0']
        diff=np.abs(ggnull.Value-ggnull0.Value)>.01
        summ=pd.concat([ggnull,ggnull0],axis=1)[diff]
        
        print('summ',len(summ))
        alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
        alpha.name='alpha'
        alpha=alpha.reset_index()
        
        power, fail=powerMat(monteCarlo(L,sigName,0,0,parms['H01'],ebb,var),alpha)

        muSD=np.sqrt(np.sum(muRange**2))
        epsSD=np.sqrt(np.sum(epsRange**2))
        muAVG=np.mean(muRange)
        epsAVG=np.mean(epsRange)
        
        epsMu=[(eps,mu) for eps in epsRange for mu in muRange]
        epsMuSorter=partial(lambda epsAVG,epsSD,muAVG,muSD,x: ((x[0]-epsAVG)/epsSD)**2+((x[1]-muAVG)/muSD)**2,epsAVG,epsSD,muAVG,muSD)
        epsMu=sorted(epsMu,key=epsMuSorter)
        
        while len(epsMu)>0:
            eps,mu=epsMu[0]
            mc=powerMat(monteCarlo(L,sigName,eps,mu,parms['H1'],ebb,var),alpha)
            power=power.append(mc[0])
            fail=fail.append(mc[1])

            epsMu=epsMu[1:]         
            if mc[0].power.max()>850:
                power=power.append(pd.DataFrame([[t_eps,t_mu,Type,1000,'1-'+str(len(parms['Types']))] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps>eps and t_mu>mu)],columns=power.columns))
                epsMu=[(t_eps,t_mu) for (t_eps,t_mu) in epsMu if not (t_eps>eps and t_mu>mu)]
            if mc[0].power.max()<350:
                power=power.append(pd.DataFrame([[t_eps,t_mu,Type,1,'1-'+str(len(parms['Types']))] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps<eps and t_mu<mu)],columns=power.columns))
                epsMu=[(t_eps,t_mu) for (t_eps,t_mu) in epsMu if not (t_eps<eps and t_mu<mu)]
        
        power.reset_index(drop=True,inplace=True)
        fail.reset_index(drop=True,inplace=True)
        power.to_csv('raw/raw-power-'+str(parms['N'])+'-'+sigName+'.csv',index=False)
        fail.to_csv('raw/raw-fail-'+str(parms['N'])+'-'+sigName+'.csv',index=False)            
    else:
        power=pd.read_csv('raw/raw-power-'+str(parms['N'])+'-'+sigName+'.csv')
        fail=pd.read_csv('raw/raw-fail-'+str(parms['N'])+'-'+sigName+'.csv')            
        
    return(power,fail,parms)
