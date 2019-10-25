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
    N=DBRead(name+'process/N',parms,True)[0]
    smallNumCores=parms['smallNumCores']
    muEpsRange=parms['muEpsRange']
    colors=parms['colors']
        
    DBCreateFolder(name,'power',parms)

    pvalFiles=pd.Series(DBListFolder(name+'pvals',parms),name='pvalFiles')
    
    for muEps in muEpsRange:
        print('plotPower started '+str(muEps),flush=True)
                
        mu=muEps[0]
        eps=muEps[1]
        nameParm='mu:'+str(mu)+'-eps:'+str(eps)
        
        DBLog(nameParm,parms)

        fig, axs = plt.subplots(1,1,dpi=50)   
        fig.set_figwidth(20,forward=True)
        fig.set_figheight(20,forward=True)

        
        #pval=DBRead(name+'pvals/ell-0.1-mu:1-eps:10-0',parms,True)
        #pdb.set_trace()
        qq=[]
        for TypeInd in range(len(Types)):
            Type=Types[TypeInd]
            
            print('plotting '+Type,flush=True)
            
            TypeMuEpsParm=Type+'-'+nameParm
            
            pvals=[]
            for TypeFile in name+'pvals/'+pvalFiles[pvalFiles.str.slice(0,len(TypeMuEpsParm))==TypeMuEpsParm]:
                pvals+=[DBRead(TypeFile,parms,True)]
            pvals=np.concatenate(pvals)
            
            DBLog(TypeMuEpsParm+' len(pvals) '+str(len(pvals))+' minP '+str(min(pvals))+' maxP '+str(max(pvals)),parms)
            
            sm.qqplot(-np.log(pvals),scipy.stats.expon,label=TypeInd,ax=axs,line='45',color=colors[TypeInd],linestyle='-')
            
        lgnd=axs.legend()
        for TypeInd in range(len(Types)):
            lgnd.legendHandles[TypeInd].set_color(colors[TypeInd])
            lgnd.get_texts()[TypeInd].set_text(Types[TypeInd])
            lgnd.legendHandles[TypeInd]._legmarker.set_markersize(6)
            
        axs.axhline(y=-np.log(.01))
        axs.axvline(x=scipy.stats.expon.ppf(.5))
        
        fig.savefig(local+name+'power/c:'+str(muEps[0])+'-n_assoc:'+str(muEps[1])+'.png',bbox_inches='tight')
        DBUpload(name+'power/c:'+str(muEps[0])+'-n_assoc:'+str(muEps[1])+'.png',parms,False)
        
        print('plotPower finished '+str(muEps),flush=True)

    return()

