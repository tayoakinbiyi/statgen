import numpy as np
import pdb
from pathlib import Path
import scipy.linalg 
import gc

def genL(parms):
    gc.collect()
    fname='L/L_'+str(int(100*parms['rho1']))+'_'+str(int(100*parms['rho2']))+'_'+str(int(100*parms['rho3']))+'_'+ \
        str(parms['N'])+'_'+str(parms['eps'])+'.csv'
    my_file = Path(fname)
    if not my_file.is_file():
        cov=np.identity(parms['N'])
        cov[:parms['eps'],:parms['eps']]=parms['rho1']*np.ones(parms['eps'])+(1-parms['rho1'])*np.identity(parms['eps'])
        n_3=int(np.floor(parms['eps']+(parms['N']-parms['eps'])/2))
        cov[parms['eps']:n_3,parms['eps']:n_3]=parms['rho3']*np.ones(n_3-parms['eps'])+(1-parms['rho3'])*np.identity(n_3-parms['eps'])
        cov[parms['eps']:,:parms['eps']]=parms['rho2']
        cov[:parms['eps'],parms['eps']:]=parms['rho2']
        L=scipy.linalg.cholesky(cov,overwrite_a=True, check_finite=False)
        np.savetxt(fname, L, delimiter=",")
    
def getL(parms):
    fname='L/L_'+str(int(100*parms['rho1']))+'_'+str(int(100*parms['rho2']))+'_'+str(int(100*parms['rho3']))+'_'+ \
        str(parms['N'])+'_'+str(parms['eps'])+'.csv'
    my_file = Path(fname)
    if my_file.is_file():
        return(np.loadtxt(fname, delimiter=","))
    else:
        raise Error("can't find L")

