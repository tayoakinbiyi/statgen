import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt

def myHist(x,title,myZip=None):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    axs.hist(x,bins=40)
    axs.set_title(title+' mu: '+('%.3E' % np.mean(x))+' std: '+('%.3E' % np.std(x)))
    
    fig.savefig('diagnostics/'+title+'.png')
    plt.close('all') 

    if not myZip is None:
        out=x.reshape(-1,1)
        myZip.writestr(title,'\n'.join(map(lambda x:','.join(map(str,x)),out.tolist()))) 
        
    return()
