
def genNullStats(parms):
    H0=parms['H0']
    maxH0Block=float(parms['maxH0Block'])
    statNames=parms['statNames']
    pvalMax=parms['pvalMax']
    scratchDir=parms['scratchDir']
        
    mu=0
    eps=0
    freq=0
    
    for i in range(int(np.ceiling(H0/maxH0Block))):
        blockSize=min(H0-i*maxH0Block,maxH0Block)
        
        z=np.matmul(L,norm.rvs([N,blockSize]).astype('float16')).astype('float16').T # H0 x N
        
        t_stats,t_fail=fitStats(ggnullDat,ghcDat,sigmaHat,z,parms) # H0 X |X|
        
        if i==0:
            stats=t_stat.apply(lambda x: x.sort_values(),axis=0).iloc[-int(3*pvalMax*blockSize):,:]
            failAvg=t_fail.sum(axis=0)
            failAll=(t_fail==0).sum(axis=0)
        else:
            stats=pd.concat([stats,t_stat],axis=0).apply(lambda x: x.sort_values(),axis=0).iloc[-int(
                3*pvalMax*(i+1)*blockSize):,:].reset_index()
            failAvg=failAvg+t_fail.sum(axis=0)
            failAll=failAvg+(t_fail==0).sum(axis=0)
            
    stats=stats.iloc[-int(pvalMax*):,:]
    stats=pd.concat([stats,pd.DataFrame({stat:-np.inf for stat in statNames})],axis=0)
    
    failAvg/=float(H0)
    failAll/=float(H0)
    
    fail=pd.concat([
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'type':'avg',**failAvg.to_dict()},index=[0]),
        pd.DataFrame({'mu':mu,'eps':eps,'freq':freq,'type':'all',**failAll.to_dict()},index=[0])],
        axis=0).reset_index()

    stats.to_csv(scratchDir+'stats-'+str(mu)+'-'+str(eps)+'-'+freq+'.csv',index=False)
    fail.to_csv(scratchDir+'fail-'+str(mu)+'-'+str(eps)+'-'+freq+'.csv',index=False)    