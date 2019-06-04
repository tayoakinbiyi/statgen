import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pdb

def makeL(parms,sig):
    sigName=parms['sigName']+'/'
    N=parms['N']
    Rpath=parms['Rpath']
    path=parms['path']
    
    if not os.path.isdir(path+sigName):
        os.mkdir(path+sigName)
        os.mkdir(path+sigName+'/ebb')
        os.mkdir(path+sigName+'/gbj')
        subprocess.run(['ln','-s',Rpath,path+sigName+'/R'])

    U,D,Vt=np.linalg.svd(sig)
    L=np.matmul(U,np.diag(np.sqrt(D)))
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(7,forward=True)
    fig.set_figheight(7,forward=True)
    off_diag=sig[np.triu_indices(N,1)].flatten()  
    axs.hist(off_diag,bins=np.linspace(-1,1,100))
    fig.suptitle(sigName)
    fig.savefig(path+sigName+'off_diag_hist.png',bbox_inches='tight')
    plt.close()    
    
    np.savetxt(path+sigName+'ebb/pairwise_cors.csv',off_diag,delimiter=',')   
    
    print('Finished MakeL')

    return(L)