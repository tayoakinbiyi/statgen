import numpy as np
import pandas as pd

def genDataSet(sigma,V_s,parms):
    K=parms['K']
    N=parms['N']
    eps=parms['eps']
    freq=parms['freq']
    mu=parms['mu']

    scratchDir+parms['scratchDir']
    
    snpData=pd.read_csv(scratchDir+'snpData.csv')
    snpChr=parms['snpChr']

    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitChr=parms['traitChr']

    numEps=int(eps*N)
    numP=int(freq*K)


    zeta=[]
    for snp in snpChr:
        snpLen=sum(snpData['chr']==snp)
        
        lam=np.abs(np.matmul(L,norm.rvs(size=[N,snpLen]).astype('float16')).astype('float16')) # N x snpLen
        lamCut=np.argsort(-lam).astype('int16')[numEps-1,:].flatten() # snpLen
        
        zeta+=[random.sample(range(snpLen),numP)] # snpLen
        
        lamCut[set(range(snpLen))-set(zeta[-1])]=-1
        lam=(lam>=lamCut) # N x snpLen
        
        theta=np.zeros([N,snpLen]).astype('float16') # N x snpLen
        theta[lam]=mu
        
        theta=theta.T.flatten().reshape(-1,1) # K*snpLen x 1
        U,D,Ut=np.lingalg.svd(np.kron(V_s,sigma))
        
        minD=min(D)
        if minD<0:
            print(snp+' min D '+minD,flush=True)
            D-=minD
            
        D=np.sqrt(D)
        L=np.matmul(U,D)
        
        z=(theta+np.matmul(L,norm.rvs([N*snpLen,1]).astype('float16')).astype('float16')).reshape(K,-1) # snpLen x N
        
        for trait in traitChr:
            pd.DataFrame(z[:,traitData['chr']==trait],index=pd.MultiIndex.from_tuples(snpData[snpData['chr']==snp].values.tolist(),
                    names=['chr','Mbp']),columns=pd.MultiIndex.from_tuples(traitData[traitData['chr']==trait].values.tolist(),
                    names=['trait','chr','Mbp']),dtype='float16').to_csv(scratchDir+'null-z-'+snp+'-'+trait+'.csv')

    zeta=[x for y in zeta for x in y]
    
    np.savetxt(scratchDir+'zeta-'+mu+'-'+p+'-'+eps+'.csv',zeta,delimiter=',')
    return(zeta,z)