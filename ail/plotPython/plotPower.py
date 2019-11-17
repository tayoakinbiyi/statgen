from ail.opPython.DB import *
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm

import pandas as pd
import numpy as np

def plotPower(parms):
    name=parms['name']
    local=parms['local']
    Types=parms['Types']
    smallNumCores=parms['smallNumCores']
    muEpsRange=[[0,0]]+parms['muEpsRange']
    colors=parms['colors']
        
    DBCreateFolder('power',parms)

    pvalFiles=pd.Series(DBListFolder('pvals',parms),name='pvalFiles')
    
    for ind in range(len(muEpsRange)):
        print('plotPower started '+str(ind),flush=True)
                
        muEps=muEpsRange[ind]
        mu=muEps[0]
        eps=muEps[1]
        
        fig, axs = plt.subplots(1,1,dpi=50)   
        fig.set_figwidth(20,forward=True)
        fig.set_figheight(20,forward=True)
        
        for TypeInd in range(len(Types)):
            Type=Types[TypeInd]
            
            print('plotting '+Type,flush=True)
            
            TypeMuEpsParm=Type+'-'+str(ind+2)
            
            pvals=DBRead('pvals/'+Type,parms)
            
            DBLog(TypeMuEpsParm+' len(pvals) '+str(len(pvals))+' minP '+str(min(pvals))+' maxP '+str(max(pvals)),parms)
            
            x=-np.log10(np.arange(1,len(pvals)+1)/(1+len(pvals)))
            y=-np.log10(np.sort(np.array(pvals)))
            axs.scatter(x,y,label=Type,color=col[Type])
            axs.plot([0,max(max(x),max(y))], [0,max(max(x),max(y))], ls="--", c=".3")        
            
        lgnd=axs.legend()
        for TypeInd in range(len(Types)):
            lgnd.legendHandles[TypeInd].set_color(colors[TypeInd])
            lgnd.get_texts()[TypeInd].set_text(Types[TypeInd])
            lgnd.legendHandles[TypeInd]._legmarker.set_markersize(6)
                    
        fig.savefig('power/c:'+str(muEps[0])+'-n_assoc:'+str(muEps[1])+'.png',bbox_inches='tight')
        
        print('plotPower finished '+str(muEps),flush=True)

    return()

