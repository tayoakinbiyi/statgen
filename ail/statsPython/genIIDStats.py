import numpy as np
from ail.opPython.DB import *
from ail.statsPython.MCMCStats import *
from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED, wait
from multiprocessing import cpu_count
import pdb
from scipy.stats import norm, beta

def genIIDStats(parms,Types,corr):    
    IIDReps=parms['IIDReps']
    name=parms['name']
    numCores=parms['numCores']
    numIIDSegments=parms['numIIDSegments']
    
    print('loading ELLAll',flush=True)

    ELLAll=DBRead(name+'ELL/ELL-'+corr,parms,True)
    LZCorr=DBRead(name+'LZCorr/LZCorr-'+corr,parms,True)
    offDiag=DBRead(name+'offDiag/offDiag-'+corr,parms,True)

    print('loaded ELLAll',flush=True)

    numBlocks=int(np.ceil(numIIDSegments/numCores))

    count=0
    for block in range(numBlocks):
        futures=[]     
        with ProcessPoolExecutor(numCores) as executor:                
            for segment in range(numCores):    
                count+=1
                futures+=[executor.submit(genIIDStatsHelp,corr+'-'+str(count),parms,ELLAll,LZCorr,offDiag,count)]

            wait(futures,return_when=ALL_COMPLETED)
        
    return()
    
def genIIDStatsHelp(nameParm,parms,ELLAll,LZCorr,offDiag,segment):
    IIDReps=parms['IIDReps']
    numCores=parms['numCores']
    numIIDSegments=parms['numIIDSegments']
    name=parms['name']
    
    randomState=segment*100
    
    N=LZCorr.shape[1]

    z=-np.sort(-np.abs(np.matmul(norm.rvs(size=[int(np.ceil(IIDReps/numIIDSegments)),N],random_state=randomState),LZCorr.T)))   
    pval=2*norm.sf(np.abs(z))
    
    corr=np.corrcoef(np.concatenate(df,axis=1),rowvar=False)
            
    DBWrite(makePSD(corr),name+'LZCorr/iid-'+nameParm,parms,True)
    
    #MCMCStats(z,pval,nameParm,parms,'iid/',ELLAll,offDiag)
    
    return()
