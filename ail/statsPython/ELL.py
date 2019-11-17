import pandas as pd
import numpy as np
import pdb
from ail.opPython.DB import *

def ELL(pval,nameParm,dParm,parms,ELLAll):
    name=parms['name']
    logName=parms['logName']
    
    stats=[]
    indexOff=0
    pct=0
    N=pval.shape[1]
    d=int(dParm*N)
    
    for k in range(d):
        ELL=ELLAll[k]
        sortOrd=np.argsort(pval[:,k])
        loc=ELL['binEdge'].searchsorted(pval[sortOrd,k]).flatten()

        outside=sum(pval[:,k]<ELL['binEdge'].min())
        pct+=outside
        indexOff+=1.0*(outside>0)

        stats+=[ELL['ell'].iloc[np.minimum(loc,len(ELL)-1)].iloc[np.argsort(sortOrd)].values.reshape(-1,1)]

    stats=-np.amin(np.concatenate(stats,axis=1),axis=1)
    DBLog('ELL-'+str(dParm)+' '+nameParm+'\tlen:min:max \t'+str(len(stats))+' : '+str(min(stats))+' : '+str(max(stats))+ \
          '\n# indices >=1 off : '+str(int(indexOff))+' of '+str(d)+ \
          '\t\t% indices off : '+str(np.round(100*pct/(d*pval.shape[0]),2)),parms)
    
    stats=pd.DataFrame({'Type':'ELL-'+str(dParm),'Value':stats})
    
    print('ELL-'+str(dParm)+' '+nameParm,flush=True)
        
    return(stats)