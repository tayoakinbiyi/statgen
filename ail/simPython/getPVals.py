import pandas as pd
import numpy as np
from scipy.stats import norm

# this function estimates the pval for each test stat for each snp
def getPVals(L,sigma,zeta,ggnullDat,ghcDat,parms):
    H0=parms['H0']
    mu=parms['mu']
    eps=parms['eps']
    eqtlFreq=parms['eqtlFreq']
    maxH0Block=float(parms['maxH0Block'])
    statNames=parms['statNames']
    pvalMax=parms['pvalMax']
    snpChr=parms['snpChr']
    cpus=parms['cpus']
    scratchDir=parms['scratchDir']
    
    stats.to_csv(scratchDir+'stats-'+str(mu)+'-'+str(eps)+'-'+freq+'.csv',index=False)
    fail.to_csv(scratchDir+'fail-'+str(mu)+'-'+str(eps)+'-'+freq+'.csv',index=False)    
    
    for snp in snpChr:
        z=pd.DataFrame(index=pd.MultiIndex.from_tuples(snpData[snpData['chr']==snp].values.tolist(),
            names=['chr','Mbp']),columns=pd.MultiIndex.from_tuples(traitData.values.tolist(),
            names=['trait','chr','Mbp']),dtype='float16')
        
        for trait in traitChr:
            z.loc[:,snpData['chr']==trait]=pd.read_csv(scratchDir+'null-z-'+snp+'-'+trait+'.csv',index_col=[0,1],header=[0,1,2])
        
        t_statsH1,t_failH1=fitStats(ggnullDat,ghcDat,sigmaHat,z,parms) # snpLen X |X|
        
        with ProcessPoolExecutor(cpus) as executor: 
            for stat in statNames:
                futures.append(executor.submit(
                    lambda statsH0,statsH1,stat: (statsH0.searchsorted(statsH1).values.flatten()/float(H0),stat,True),
                    statsH0[stat],statsH1Eqtl[stat],stat))                          

                futures.append(executor.submit(
                    lambda statsH0,statsH1,stat: (statsH0.searchsorted(statsH1).values.flatten()/float(H0),stat,False),
                    statsH0[stat],statsH1NoEqtl[stat],stat))                          

            for f in as_completed(futures):
                ans=f.result()

                j=0
                pval=ans[j];j+=1
                stat=ans[j];j+=1
                eqtl=ans[j];j+=1

                pvalsEqtl[stat]=pval
    
def getPValsHelp(statsH0,statsH1,stat:
    val=statsH0.searchsorted(statsH1)
    val[val==len(statsH0)-1]=H0
    val=val.values.flatten()/float(H0)
    
    return(val,
                 ,stat,True),
                    statsH0[stat],statsH1Eqtl[stat],stat)        
    statsH1Eqtl=stats[zeta==1].apply(lambda x: x.sort_values(),axis=0).reset_index()
    statsH1NoEqtl=stats[zeta==0].apply(lambda x: x.sort_values(),axis=0).reset_index()
    
    futures=[]
    pvalsEqtl=pd.DataFrame(columns=statNames,index=range(len(statsH1Eqtl)))
    pvalsNoEqtl=pd.DataFrame(columns=statNames,index=range(len(statsH1NoEqtl)))
        
            
    pvalsEqtl.index=0
    pvalsEqtl=pd.DataFrame({'mu':mu,'eps':eps,'freq':freq},index=[0]).merge(pvalsEqtl,left_index=True,
        right_index=True).reset_index()

    pvalsNoEqtl.index=0
    pvalsNoEqtl=pd.DataFrame({'mu':mu,'eps':eps,'freq':freq},index=[0]).merge(pvalsNoEqtl,left_index=True,
        right_index=True).reset_index()

    fail=pd.concat(
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',True,'type':'avg',**failH0.mean(axis=0).to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',True,'type':'all',**(failH0==0).mean(axis=0).to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',False,'type':'avg',**failH1.mean(axis=0).to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'H0',False,'type':'all',**(failH1==0).mean(axis=0).to_dict()},index=[0]),
        axis=0).reset_index()
    
    return(pvalsEqtl,pvalsNoEqtl,failH0,failH1)

