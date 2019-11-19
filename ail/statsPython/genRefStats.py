import numpy as np
from ail.opPython.DB import *
from ail.statsPython.MCMCStats import *
from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED, wait
from multiprocessing import cpu_count
import pdb
from scipy.stats import norm, beta

def genRefStats(parms,Types,corrName):    
    numCores=parms['numCores']
    RefReps=parms['RefPerCore']
    
    print('loading ELLAll',flush=True)

    L=DBRead('LZCorr/'+corrName,parms)
    corr=np.matmul(L,L.T)
    N=corr.shape[0]
    offDiag=corr[np.triu_indices(N,1)].flatten()
    ELLAll=DBRead('ELL/ELL-'+str(N),parms)

    print('loaded ELLAll',flush=True)
    DBCreateFolder('ref',parms)

    seeds=random.sample(range(int(1e6)),numCores)

    futures=[]
    with ProcessPoolExecutor(numCores) as executor:                
        for core in range(numCores):    
            futures+=[executor.submit(genRefStatsHelp,parms,ELLAll,L,offDiag,core,Types,seeds[core])]

        mat=[]
        for f in wait(futures,return_when=ALL_COMPLETED)[0]:
            mat+=[f.result()]
        
        mat=pd.concat(mat,axis=0)
        
    for Type in mat['Type'].drop_duplicates().values:
        DBWrite(mat[mat['Type']==Type].sort_values(by='Value')['Value'].reset_index(),'ref/'+Type,parms)
        
    return()
    
def genRefStatsHelp(parms,ELLAll,L,offDiag,segment,Types,seed):
    RefReps=parms['RefReps']
    numCores=parms['numCores']

    RefPerCore=int(np.ceil(RefReps/numCores))
    
    N=L.shape[0]

    z=-np.sort(-np.abs(np.matmul(norm.rvs(size=[RefPerCore,N],random_state=seed),L.T)))   
    pval=2*norm.sf(np.abs(z))        
    
    return(MCMCStats(z,pval,str(segment),parms,ELLAll,offDiag,Types))
