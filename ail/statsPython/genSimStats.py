from ail.opPython.DB import *
from ail.statsPython.MCMCStats import *
from scipy.stats import norm, beta

import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait,ALL_COMPLETED

def genSimStats(parms,Types,corrName,snpChr):
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    numSimStatSegments=parms['numSimStatSegments']
                    
    print('loading ELLAll',flush=True)

    L=DBRead('LZCorr/'+corrName,parms)
    corr=np.matmul(L,L.T)
    N=corr.shape[0]
    offDiag=corr[np.triu_indices(N,1)].flatten()
    ELLAll=DBRead('ELL/ELL-'+str(N),parms)

    print('loaded ELLAll',flush=True)
    DBCreateFolder('stats',parms)

    for snp in snpChr:
        df=[]
        for trait in traitChr:
            df+=[DBRead('score/'+snp+'-'+trait,parms)]

        z=-np.sort(-np.abs(np.concatenate(df,axis=1)))
        pval=2*norm.sf(np.abs(z))

        segLen=int(np.ceil(len(z)/numSimStatSegments))
        futures=[]
        with ProcessPoolExecutor(numCores) as executor: 
            for segment in range(numSimStatSegments):
                snpRange=range(segment*segLen,min((segment+1)*segLen,len(z)))
                futures+=[executor.submit(MCMCStats,z[snpRange],pval[snpRange],snp+'-'+str(segment),parms,ELLAll,
                    offDiag,Types)]

            mat=[]
            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                mat+=[f.result()]

            mat=pd.concat(mat,axis=0)

        for Type in mat['Type'].drop_duplicates().values:
            DBWrite(mat[mat['Type']==Type]['Value'].reset_index(),'stats/'+Type,parms)

    return()
    