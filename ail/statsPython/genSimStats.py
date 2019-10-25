from ail.opPython.DB import *
from ail.statsPython.MCMCStats import *
from scipy.stats import norm, beta

import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait,ALL_COMPLETED

def genSimStats(parms,snp,Types,corr):
    traitChr=parms['traitChr']
    name=parms['name']
    numCores=parms['numCores']
    numSimStatSegments=parms['numSimStatSegments']
    muEpsRange=parms['muEpsRange']
    wald=parms['wald']
        
    DBSyncLocal(name+'process',parms)
            
    print('loading ELLAll',flush=True)

    nameParms=['mu:'+str(x[0])+'-eps:'+str(x[1]) for x in muEpsRange]
    if len(nameParms)==0:
        nameParms=['']
        
    ELLAll=DBRead(name+'ELL/ELL-'+corr,parms,True)
    offDiag=DBRead(name+'offDiag/offDiag-'+corr,parms,True)
    
    scoreFiles=pd.Series(DBListFolder(name+'score',parms),name='scoreFiles')
    
    print('loaded ELLAll',flush=True)

    for nameParm in nameParms:
        print(nameParm,flush=True)
        if len(muEpsRange)==0:
            fileParm=''
            nameParm=snp
        else:
            fileParm='-'+nameParm

        if sum(scoreFiles.str.contains('p-'+snp))==0:
            continue

        t=[]
        for trait in traitChr:
            t+=[DBRead(name+'score/p-'+snp+'-'+trait+fileParm,parms,True)]

        if wald:
            z=-np.sort(-np.abs(np.concatenate(t,axis=1)))
            pval=2*norm.sf(np.abs(z))
        else:
            pval=np.sort(np.concatenate(t,axis=1))
            z=norm.ppf(pval/2)

        segLen=int(np.ceil(len(z)/numSimStatSegments))
        futures=[]
        with ProcessPoolExecutor(numCores) as executor: 
            for segment in range(numSimStatSegments):
                snpRange=range(segment*segLen,min((segment+1)*segLen,len(z)))
                futures+=[executor.submit(MCMCStats,z[snpRange],pval[snpRange],nameParm+'-'+str(segment),parms,'stats/',ELLAll,offDiag)]

            wait(futures,return_when=ALL_COMPLETED)                        

    return()
    