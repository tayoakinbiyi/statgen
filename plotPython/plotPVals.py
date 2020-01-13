from opPython.DB import *
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pdb

def plotPVals(parms,snpChr):
    name=parms['name']
    local=parms['local']
    Types=parms['Types']
    N=DBRead(name+'process/N',parms,True)[0]
    snpData=DBRead(name+'process/snpData',parms,toPickle=True)
   
    DBCreateFolder(name,'pvalPlot',parms)

    pvalFiles=pd.Series(DBListFolder(name+'pvals',parms),name='pvalFiles')

    for Type in Types:
        numChr=len(snpChr)
        fig, axs = plt.subplots(1,1,dpi=50,tight_layout=True)   
        fig.set_figwidth(20*numChr,forward=True)
        fig.set_figheight(20,forward=True)

        snpData=snpData[snpData['chr'].isin(snpChr)]
        snpData.insert(0,'ord',snpData['chr'].str.slice(3).astype(int))
        snpData=snpData.sort_values(by='ord')
        snpData.insert(0,'cum',snpData['Mbp'].cumsum()/1e6)

        for snp in range(len(snpChr)):
            pvals=[]
            for snpFile in name+'pvals/'+pvalFiles[pvalFiles.str.contains(Type+'-'+snpChr[snp]+'-')]:
                pvals+=[DBRead(snpFile,parms,True)]
            
            pvals=-np.log10(np.concatenate(pvals))

            axs.scatter(snpData.loc[snpData['chr']==snpChr[snp],'cum'].values.flatten(),pvals,label=Type,s=1)
            
        axs.set_xlabel('Mbp')
        axs.set_ylabel('pval')
        axs.legend()
    
        for snp in range(len(snpChr)-1):
            axs.axvline(x=np.amax(snpData.loc[snpData['chr']==snpChr[snp],'cum'].values.flatten()),color='b')
        
        midChr=snpData.groupby('chr').mean().reset_index()  
        axs.set_xticks(midChr['cum'])
        axs.set_xticklabels(midChr['chr'])
        axs.tick_params(axis='both',labelsize=12)
        
        fig.savefig(local+name+'pvalPlot/'+Type+'.png',bbox_inches='tight')
        DBUpload(name+'pvalPlot/'+Type+'.png',parms,False)
        pdb.set_trace()
        print('plotPVals finished '+Type,flush=True)

    return()

        
