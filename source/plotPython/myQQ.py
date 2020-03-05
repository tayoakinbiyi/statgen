import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt

def myQQ(x,y,title,ylabel='observed',xlabel='theoretical'):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    labMax=max(max(x),max(y))
    labMin=min(min(x),min(y))
    labMax+=.1*(labMax-labMin)
    labMin-=.1*(labMax-labMin)

    axs.set_xlim([labMin,labMax])
    axs.set_ylim([labMin,labMax])
    axs.scatter(x,y,c='.1')
    axs.plot([labMin,labMax], [labMin,labMax], ls="--", c=".3")  
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title+' mu: '+('%.3E' % np.mean(y))+' std: '+('%.3E' % np.std(y)))
    
    fig.savefig('diagnostics/'+title+'.png')
    plt.close('all') 

    return()
