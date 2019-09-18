from ail.simPython.MCMCStats import *

from ail.statsPython.ghc import *
from ail.statsPython.gbj import *

from ail.opPython.DB import *

import pandas as pd

def getPVals(z,mu,eps,parms):
    name=parms['name']
    cpus=parms['cpus']

    N=DBRead(name+'process/N',parms,toPickle=False)
    
    z=-np.sort(-np.abs(z))[:,0:int(N/2)]
    Reps=len(z)

    numSegs=int(np.ceil(Reps/np.ceil(Reps/cpus)))
    for i in range(numSegs):
        np.savetxt(path+'sim/z-'+str(i)+'.csv',z[i*int(np.ceil(Reps/cpus)):min((i+1)*int(np.ceil(Reps/cpus)),Reps)],delimiter=',')
    
    pvals=pd.DataFrame()
    pvals=pvals.append(MCMCStats(z,parms))
    pvals=pvals.append(ghc(numSegs,parms))
    pvals=pvals.append(cpma(numSegs,parms))
    pvals=pvals.append(gbj(numSegs,str(parms['mu'])+'-'+str(parms['epsilon']),parms)) 
    
    pvals.insert(0,'mu',mu)
    pvals.insert(0,'eps',eps)

    return(pvals)

