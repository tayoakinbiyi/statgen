from ail.opPython.DB import *
from ail.statsPython.genStats import *
from ail.statsPython.MCMCStats import *

import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED

def genObsStats(parms):
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    name=parms['name']
    numCores=parms['numCores']
        
    DBSyncLocal(name+'process',parms)
    
    DBCreateFolder(name,'stats',parms)

    snpData=DBRead(name+'process/snpData',parms,True)
    traitData=DBRead(name+'process/traitData',parms,True)
        
    print('loading ELLAll',flush=True)

    futures=[]
    with ProcessPoolExecutor(numCores) as executor: 
        for snp in snpChr:
            z=[]
            for trait in traitChr:
                if trait==snp:
                    continue
                z+=[DBRead(name+'score/p-'+snp+'-'+trait+zParm,parms,toPickle=True)]
            z=-np.sort(-np.abs(np.concatenate(z,axis=1)))

            ELLAll=DBRead(name+'ELL/ELL-'+snp,parms,True) 
            offDiag=DBRead(name+'offDiag/offDiag-'+snp,parms,True)

            futures+=[executor.submit(MCMCStats,z,nameParm,parms,N,'stats/',ELLAll,offDiag)]

        wait(futures,return_when=ALL_COMPLETED)

    return()
