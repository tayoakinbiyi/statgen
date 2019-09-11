import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pdb

from ail.opPython.DB import *

def makeL(parms,sig):
    name=parms['name']
    N=parms['N']
    
    U,D,Vt=np.linalg.svd(sig)
    L=np.matmul(U,np.diag(np.sqrt(D)))
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(7,forward=True)
    off_diag=sig[np.triu_indices(N,1)].flatten()  
    axs.hist(off_diag,bins=np.linspace(-1,1,100))
    fig.savefig(plotsDir+'off_diag_hist.png',bbox_inches='tight')
    plt.close()    
    
    DBWrite(off_diag,name+'process/pairwise_cors',parms,toPickle=true)   
    
    print('Finished MakeL')

    return(L)