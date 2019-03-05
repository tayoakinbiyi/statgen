from monteCarlo import *
from norm_sig import *
from makeProb import *

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
    N=parms['N']
    L=np.linalg.cholesky(parms['sig'])
    
    z=np.matmul(L.T,np.random.normal(0,1,size=(N,max(parms['H0'],parms['H1'],parms['H01'])))).T
    empSig=np.corrcoef(z,rowvar=False)
    sig_tri=empSig[np.triu_indices(N,1)].flatten() #np.array([0]*int(N*(N-1)/2)) 
    
    t=time.time()
    makeProb(L,np.unique(parms['muRange']).round(3),np.unique(parms['epsRange']).round().astype(int),40,sig_tri,N,parms['sigName'])
    print('makeProb '+str((time.time()-t)/60))
        
    if parms['new']:
        alpha,_=monteCarlo(parms['H0'],{**parms,'sig_tri':sig_tri},0,0,z[0:parms['H0'],:])
        alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
        alpha.name='alpha'
        alpha=alpha.reset_index()
        pdb.set_trace()
        
        power, fail=powerMat(monteCarlo(parms['H01'],{**parms,'sig_tri':sig_tri},0,0,z[0:parms['H01'],:],ebb),alpha)
        
        muSD=np.sqrt(np.sum(np.unique(parms['muRange'])**2))
        epsSD=np.sqrt(np.sum(np.unique(parms['epsRange'])**2))
        muAVG=np.mean(np.unique(parms['muRange']))
        epsAVG=np.mean(np.unique(parms['epsRange']))
        
        epsMu=[(eps,mu) for eps in np.unique(parms['epsRange'].round().astype(int)) for mu in np.unique(parms['muRange']).round(3)]
        epsMu=sorted(epsMu,key=lambda x: ((x[0]-epsAVG)/epsSD)**2+((x[0]-muAVG)/muSD)**2)
        
        while len(epsMu)>0:
            eps,mu=epsMu[0]
            mc=powerMat(monteCarlo(parms['H1'],{**parms,'sig_tri':sig_tri},mu,eps,z[0:parms['H1'],:],ebb),alpha)
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
