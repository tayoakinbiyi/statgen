import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from zipfile import ZipFile

def myQQ(x,y,xlabel='observed',ylabel='theoretical'):
    title=ylabel+' ~ '+xlabel
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    labMax=max(max(x),max(y))
    labMin=min(min(x),min(y))
    labMax+=.1*(labMax-labMin)
    labMin-=.1*(labMax-labMin)

    axs.set_xlim([labMin,labMax])
    axs.set_ylim([labMin,labMax])
    axs.scatter(x,y,s=.1,color='k')
    axs.plot([labMin,labMax], [labMin,labMax], ls="--")  
    axs.set_xlabel(xlabel+' mu: '+('%.3E' % np.mean(x))+' std: '+('%.3E' % np.std(x)))
    axs.set_ylabel(ylabel+' mu: '+('%.3E' % np.mean(y))+' std: '+('%.3E' % np.std(y)))
    axs.set_title(title)
    
    with ZipFile('diagnostics/'+title+'.zip','w') as myZip:
        out=np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)
        myZip.writestr(title,'\n'.join(map(lambda x:'\t'.join(map(str,x)),out.tolist()))) 
    
    fig.savefig('diagnostics/'+title+'.png')
    plt.close('all') 

    return()
