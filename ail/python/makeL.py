import numpy as np
import matplotlib.pyplot as plt
import os

def makeL(parms):
    sig=parms['sig']
    sigName=parms['sigName']
    N=parms['N']
    
    U,D,Vt=np.linalg.svd(sig)
    L=np.matmul(U,np.diag(np.sqrt(D)))
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(7,forward=True)
    off_diag=sig[np.triu_indices(N,1)].flatten()  
    axs.hist(off_diag,bins=np.linspace(-1,1,100))
    fig.suptitle(sigName)
    
    np.savetxt('ebb/'+sigName+'/pairwise_cors.csv',off_diag,delimiter=',')
    
    if not os.path.isdir('ebb/'+sigName):
        os.mkdir('ebb/'+sigName)

    fig.savefig('ebb/'+sigName+'/pairwise_cors.png',bbox_inches='tight')
    plt.close()    

    return(L)