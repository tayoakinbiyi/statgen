from opPython.DB import *
from scipy.stats import norm, beta

import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait,ALL_COMPLETED

def genGBJPVals(parms,snpChr):
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    numServers=parms['numServers']
                    
    print('loading ELLAll',flush=True)

    L=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
    corr=np.matmul(L,L.T)
    N=corr.shape[0]
    offDiag=corr[np.triu_indices(N,1)].flatten()
    ELLAll=DBRead('ELL/ELL-'+str(N),parms)

    print('loaded ELLAll',flush=True)
    
    if input('1 to reset gbj, <enter> to continue : ')=='1':
        DBCreateFolder('gbj',parms)

    for snp in snpChr:
        df=[]
        for trait in traitChr:
            df+=[np.loadtxt('score/waldStat-'+str(snp)+'-'+str(trait),delimiter='\t')]

        zAll=-np.sort(-np.abs(np.concatenate(df,axis=1)))

        serverLen=int(np.ceil(len(zAll)/numServers))
        for server in range(numServers):
            serverRange=range(server*serverLen,min((server+1)*serverLen,len(zAll)))
            if len(serverRange)==0:
                continue
            
            if os.path.exists('gbj/gbjPvals-'+str(snp)+'-'+str(server)):
                continue
            np.savetxt('gbj/gbjPvals-'+str(snp)+'-'+str(server),np.array([]),delimiter='\t')
            
            z=zAll[serverRange]            
            
            segLen=int(np.ceil(len(z)/numCores))
            futures=[]
            with ProcessPoolExecutor(numCores) as executor: 
                for core in range(numCores):
                    snpRange=range(core*segLen,min((core+1)*segLen,len(z)))
                    if len(snpRange)==0:
                        continue

                    #gbj(z[snpRange],str(snp)+'-'+str(core),parms,offDiag)
                    futures+=[executor.submit(gbj,z[snpRange],parms,offDiag)]
                    print('gbj - snp: '+str(snp)+' - server: '+str(server)+' - core: '+str(core),flush=True)

                mat=[]
                for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                    mat+=[f.result()]

                mat=pd.concat(mat,axis=0)

            for Type in mat['Type'].drop_duplicates().values:
                np.savetxt('gbj/'+Type+'-'+str(snp)+'-'+str(server),mat.loc[mat['Type']==Type,'Value'],delimiter='\t')
          
    signal=input('1 to pool, <enter> to exit : ')
    if signal=='':
        return()
    
    Types=['bj','gbj','hc','ghc','minP']
    
    for snp in snpChr:
        for Type in Types:
            df=[]
            for server in range(numServers):
                df+=[np.loadtxt('gbj/'+Type+'-'+str(snp)+'-'+str(server),delimiter='\t')]

            np.savetxt('pvals/'+Type+'-'+str(snp),np.concatenate(df,axis=0),delimiter='\t')
            
    return()
    