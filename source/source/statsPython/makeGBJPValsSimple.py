from opPython.DB import *
from scipy.stats import norm, beta
from statsPython.gbj import *

import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait,ALL_COMPLETED

def makeGBJPVals(parms):
    traitChr=parms['traitChr']
    numCores=parms['numCores']
    muEpsRange=[[0,0]]+parms['muEpsRange']
                    
    L=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
    corr=np.matmul(L,L.T)
    N=corr.shape[0]
    offDiag=corr[np.triu_indices(N,1)].flatten()

    for ind in range(len(muEpsRange)):                
        muEps=muEpsRange[ind]
        mu=muEps[0]
        eps=muEps[1]
        snp=len(parms['SnpSize'])+ind

        df=[]
        for trait in traitChr:
            df+=[np.loadtxt('score/waldStat-'+str(snp)+'-'+str(trait),delimiter='\t')]

        z=-np.sort(-np.abs(np.concatenate(df,axis=1)))

        segLen=int(np.ceil(len(z)/numCores))
        futures=[]
        with ProcessPoolExecutor(numCores) as executor: 
            for core in range(numCores):
                snpRange=range(core*segLen,min((core+1)*segLen,len(z)))
                if len(snpRange)==0:
                    continue

                futures+=[executor.submit(gbj,z[snpRange],offDiag)]

            mat=[]
            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                mat+=[f.result()]

            mat=pd.concat(mat,axis=0)

        for Type in mat['Type'].drop_duplicates().values:
            np.savetxt('pvals/'+Type+'-'+str(snp),mat.loc[mat['Type']==Type,'Value'],delimiter='\t')
                      
    return()
    