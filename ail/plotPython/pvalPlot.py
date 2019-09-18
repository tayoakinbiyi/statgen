import pdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.distributions.empirical_distribution import ECDF

from ail.opPython.DB import *
from ail.simPython.makePower import *

def pvalPlot(parms):
    mu=np.round(sorted(power.mu.drop_duplicates().values.tolist()),2)
    eps=np.round(sorted(power.eps.drop_duplicates().values.tolist()),2)

    fig, axs = plt.subplots(len(mu),len(eps),dpi=50)   
    fig.set_figwidth(len(mu)*3,forward=True)
    fig.set_figheight(len(eps)*3,forward=True)
    
    power=DBRead(name+'sim/power',parms,toPickle=True)
    
    muList=power['mu'].drop_duplicates().values.flatten().tolist()
    epsList=power['eps'].drop_duplicates().values.flatten().tolist()

    pvalValues=power['Value'].drop_duplicates().sort_values().values.flatten().toList()
    logPvalValues=-np.log10(pvalValues)
    Types=power['Type'].drop_duplicates().values.flatten().toList()
    
    for mu in range(len(muList)):
        for eps in range(len(epsList)):
            for Type in Types:
                ecdf=ECDF(power.loc[(power['mu']==muList[mu])&(power['eps']==epsList[eps])&(power['Type']==Type),'Value'])(pvalValues)
                axs[mu,eps].scatter(logPvalValues,ecdf,label=Type)
                axs[mu,eps].set_xlabel('eps')
                axs[mu,eps].set_ylabel('mu')
                axs[mu,eps].legend(fontsize=20)
                axs[mu,eps].set_title('mu : '+str(mu)+'- eps : '+str(eps))
                axs[mu,eps].axvline(x=-np.log10(.01))

    fig.savefig(local+name+'plots/H1PvalPlot.png',bbox_inches='tight')
    DBUpload(name+'plots/H1PvalPlot.png',parms,toPickle=False)
    
    return()