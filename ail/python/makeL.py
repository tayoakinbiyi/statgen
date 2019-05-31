import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pdb

def makeL(parms,sig):
    sigName=parms['sigName']
    N=parms['N']
    Rpath=parms['Rpath']
    path=parms['path']
    
    os.chdir(path)
    
    if not os.path.isdir(sigName):
        os.mkdir(sigName)
        os.mkdir(sigName+'/ebb')
        os.mkdir(sigName+'/gbj')
        subprocess.run(['ln','-s',Rpath,sigName+'/R'])

    os.chdir(path+sigName)

    U,D,Vt=np.linalg.svd(sig)
    L=np.matmul(U,np.diag(np.sqrt(D)))
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(7,forward=True)
    off_diag=sig[np.triu_indices(N,1)].flatten()  
    axs.hist(off_diag,bins=np.linspace(-1,1,100))
    fig.suptitle(sigName)
    fig.savefig('off_diag_hist.png',bbox_inches='tight')
    plt.close()    
    
    np.savetxt('ebb/pairwise_cors.csv',off_diag,delimiter=',')   
    
    print('Finished MakeL')

    return(L)