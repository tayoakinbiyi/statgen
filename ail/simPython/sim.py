from python.monteCarlo import *
from python.norm_sig import *
from python.makeProb import *

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
'''
,globals:
muList=sequence from 1 to 2 with 5 increments, which is the list of mu to evaluate the power
epsilonList= sequence from 20/16000 to 500/16000 in 5 increments
pList=sequence from .0001 to .001 with 5 increments

C_power  = .01 # pval cutoff for power calculations
C_genome = 5*10-6 # pval cutoff genome wide

expr_n = vector of original expression levels of gene g across mice
snp_k is genotype vector for snp k across all mice

X = {gbj, ELL,...,GHC,...,FDR, minP}
f_x(Z,sigma) = function that returns test statistic x in X for the z scores for a given snp Z

J=50,000 # the number of null distribution reps used in power simulations
N=16,000
K=523,000





# this function estimates the power of each test stat x in X
findPower({zeta_k : k=1,...,K}, {P_kx : k=1,...,K  x in X}, C)
{
# power_x is the power of stat x 
power_x = (1/|{k : \zeta_k=1}|)\sum_{k : \zeta_k=1}I[P_kx<=C , x in X

# Type1_x is the type 1 error rate of stat x 
Type1_x = (1/|{k : \zeta_k=0}|)\sum_{k : \zeta_k=0}I[P_kx<=C] , x in X

return {power_x : x in X} , {Type1_x : x in X}

    fail=pd.concat(
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',True,'type':'avg',**failH0.mean(axis=0).to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',True,'type':'all',**(failH0==0).mean(axis=0).to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',False,'type':'avg',**failH1.mean(axis=0).to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',False,'type':'all',**(failH1==0).mean(axis=0).to_dict()},index=[0]),
        axis=0).reset_index()
    
}

# generate z scores from original AIL data and estimate correlation across genes
sigmaOrig, {Z_k : k=1,...,K}=genZScore({expr_n : n=1,...,N}, {snp_k : k =1,...,K})

# estimate pval for each stat for each snp k
{P_kx : k=1,...,K , x in X}=calcPval(sigmaOrig, {Z_k : k=1,...,K})

# eqtl_k=1 if snp k is deemed an eqtl for some gene, and 0 otherwise
eqtl_k=I[ min{P_kx : x in X} <= C_genome], k=1,...,K

# create power and type 1 error estimates
for mu, epsilon, p in muList x epsilonList x pList:
{
# generate the simulated Dataset, i.e. simulating the Z-scores from ail mouse data
{zeta_k: k=1,..,K}, hat{sigma}, {Z_k: k=1,...,K} = genDataSet(sigmaOrig, epsilon, p, mu)

# estimate pval for each stat for each snp k
{P_kx : k=1,...,K , x in X}=calcPval(sigma, {Z_k : k=1,...,K})

 # estimate the power and type 1 error for 
{power_x : x in X}, {Type1_x : x in X}=findPower({zeta_k : k=1,...,K}, {P_kx : k=1,...,K  x in X}, C_power)

{power_mu,epsilon,p,x : x in X}={power_x : x in X}
{Type1_mu,epsilon,p,x : x in X}={Type1_x : x in X}
}
'''

# this function generates a simulated origin dataset (i.e simulating the observed Z scores from 
# AIL dataset)
# as well as generating the estimated Var of z scores across traits for a given snp (i.e. sigma)
# sigma is sigma of z scores across traits for a given snp
# epsilon is percentage of traits snp is an eqtl for
# p is prob a snp is an eqtl for any trait
# mu is the mean of z scores for traits where the snp is an eqtl


    sigmaHat=genCorr('',{**files,'scratchDir':scratchDir+'null-'})
    ggnullDat,ghcDat=makeProb(L,files)
    

                                                     
def sim(parms):
    maxPower=1000
    minPower=0
    
    N=parms['N']
    L=np.linalg.cholesky(parms['sig'])  
    
    epsRange=parms['epsRange']
    muRange=parms['muRange']
    sigName=parms['sigName']
    
    t0=time.time()
    
    ggnullDat,ghcDat=makeProb(L,parms)
    print('makeProb',round((time.time()-t0),2))
    
    if parms['new']:
        alpha,_=monteCarlo(L,sigName,0,0,parms['H0'],ggnullDat,ghcDat)
        alpha=alpha.groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
        alpha.name='alpha'
        alpha=alpha.reset_index()
        
        power, fail=powerMat(monteCarlo(L,sigName,0,0,parms['H01'],ggnullDat,ghcDat),alpha)

        muSD=np.sqrt(np.sum(muRange**2))
        epsSD=np.sqrt(np.sum(epsRange**2))
        muAVG=np.mean(muRange)
        epsAVG=np.mean(epsRange)
        
        epsMu=[(eps,mu) for eps in epsRange for mu in muRange]
        epsMuSorter=partial(lambda epsAVG,epsSD,muAVG,muSD,x: ((x[0]-epsAVG)/epsSD)**2+((x[1]-muAVG)/muSD)**2,epsAVG,epsSD,muAVG,muSD)
        epsMu=sorted(epsMu,key=epsMuSorter)
        
        while len(epsMu)>0:
            eps,mu=epsMu[0]
            mc=powerMat(monteCarlo(L,sigName,eps,mu,parms['H1'],ggnullDat,ghcDat),alpha)
            power=power.append(mc[0])
            fail=fail.append(mc[1])

            epsMu=epsMu[1:]         
            if mc[0].power.max()>maxPower:
                power=power.append(pd.DataFrame([[t_eps,t_mu,Type,1000,'1-'+str(len(parms['Types']))] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps>eps and t_mu>mu)],columns=power.columns))
                fail=fail.append(pd.DataFrame([[t_eps,t_mu,Type,np.nan,np.nan] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps<eps and t_mu<mu)],columns=fail.columns))
                epsMu=[(t_eps,t_mu) for (t_eps,t_mu) in epsMu if not (t_eps>eps and t_mu>mu)]
            if mc[0].power.max()<minPower:
                power=power.append(pd.DataFrame([[t_eps,t_mu,Type,1,'1-'+str(len(parms['Types']))] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps<eps and t_mu<mu)],columns=power.columns))
                fail=fail.append(pd.DataFrame([[t_eps,t_mu,Type,np.nan,np.nan] for (t_eps,t_mu) in epsMu 
                       for Type in parms['Types'] if (t_eps<eps and t_mu<mu)],columns=fail.columns))
                epsMu=[(t_eps,t_mu) for (t_eps,t_mu) in epsMu if not (t_eps<eps and t_mu<mu)]
        
        power.reset_index(drop=True,inplace=True)
        fail.reset_index(drop=True,inplace=True)
        power.to_csv('../raw/raw-power-'+str(parms['N'])+'-'+sigName+'.csv',index=False)
        fail.to_csv('../raw/raw-fail-'+str(parms['N'])+'-'+sigName+'.csv',index=False)            
    else:
        power=pd.read_csv('../raw/raw-power-'+str(parms['N'])+'-'+sigName+'.csv')
        fail=pd.read_csv('../raw/raw-fail-'+str(parms['N'])+'-'+sigName+'.csv')            
        
    return(power,fail,parms)
