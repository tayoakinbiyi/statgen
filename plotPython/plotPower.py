from opPython.DB import *
import matplotlib.pyplot as plt
from scipy.stats import beta
import statsmodels.api as sm

import pandas as pd
import numpy as np

def plotPower(parms):
    local=parms['local']
    muEpsRange=[[0,0]]+parms['muEpsRange']
    colors=parms['colors']
    SnpSize=parms['SnpSize'][-1]
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
        
    DBCreateFolder('power',parms)

    Types=pd.Series(os.listdir('pvals'),name='Types')
    Types=Types.str.extract(r'^([0-9a-zA-Z_\.]+)-[0-9]+$').iloc[:,0].drop_duplicates().values.flatten()
    
    snpData=pd.read_csv('ped/snpData',sep='\t',header=0,index_col=None)
    
    fig, axs = plt.subplots(len(snpChr),1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10*len(snpChr),forward=True)
    
    localLevels=pd.read_csv(local+'data/local_level_results.csv',header=0,index_col=None).values
    eta=localLevels[np.argmin(np.abs(localLevels[:,0]-SnpSize)),1]
    nVec=np.arange(1,SnpSize+1)
    lowerbounds=beta.ppf(eta/2,nVec,nVec[::-1])
    upperbounds=beta.ppf(1-eta/2,nVec,nVec[::-1])                           
    
    if len(snpChr)==1:
        axs=[axs]
        
    xLoc=np.arange(1,SnpSize+1)/(1+SnpSize)
    
    exact=[]

    for ind in range(len(snpChr)):                
        muEps=muEpsRange[ind]
        mu=muEps[0]
        eps=muEps[1]
        snp=snpChr[ind]
                        
        df=pd.DataFrame(index=-np.log10(xLoc),columns=Types)
        for TypeInd in range(len(Types)):
            Type=Types[TypeInd]
            
            print('plotting '+Type,flush=True)
            pvals=np.loadtxt('pvals/'+Type+'-'+str(snp),delimiter='\t')
            
            DBLog(Type+'-'+str(snp)+' len(pvals) '+str(len(pvals))+' minP '+str(min(pvals))+' maxP '+str(max(pvals)),parms)
            df.loc[:,Type]=-np.log10(np.sort(pvals))
            
            exact+=[['mu: '+str(mu)+'- eps: '+str(eps),Type,np.mean(pvals<.05),np.mean(pvals<.01)]]
        
        bounds=pd.DataFrame({'lower':-np.log10(upperbounds),'upper':-np.log10(lowerbounds)},index=-np.log10(xLoc))
        
        mMax=max(df.index.max(),df.max().max())*1.1
        df.plot(ax=axs[ind],legend=True,xlim=[0,mMax],ylim=[0,mMax],color=colors[0:len(Types)])
        axs[ind].plot([0,mMax], [0,mMax], ls="--", c=".3")   
        axs[ind].set_title('c: '+str(mu)+'     n_assoc: '+str(eps),fontsize=20)
        bounds.plot(ax=axs[ind],legend=False,xlim=[0,mMax],ylim=[0,mMax],color='black')
                                
        print('plotPower finished '+str(mu)+'-'+str(eps),flush=True)
        
    pd.DataFrame(exact,columns=['scenario','Type','.05','.01']).to_csv('diagnostics/exact.tsv',sep='\t',index=False)
        
    fig.savefig('diagnostics/power.png',bbox_inches='tight')

    return()

