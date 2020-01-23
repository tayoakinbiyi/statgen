import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import re

def manhattanPlots(parms,B=5):
    scratchDir=parms['scratchDir']
    plotsDir=parms['plotsDir']    
    dataDir=parms['dataDir']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    smallCpu=parms['smallCpu']

    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitData['Mbp']=traitData['Mbp'].astype(float)

    snpData=pd.read_csv(scratchDir+'snpData.csv')
    snpData['Mbp']=snpData['Mbp'].astype(float)

    ail_trans=pd.read_csv(dataDir+'ail_paper-Trans.csv',header=0)
    ail_trans=ail_trans[ail_trans['eqtl_tissue']=='hip']

    trans=[ail_trans[ail_trans['target_gene_chrom']==i+1].sort_values(by='eqtl_pvalue')[
        ['target_gene','eqtl_chrom','eqtl_pos_bp']].iloc[0:B] for i in range(len(traitChr))]
    
    ail_cis=pd.read_csv(dataDir+'ail_paper-Cis.csv',header=0)
    ail_cis=ail_cis[ail_cis['tissue']=='hip']
    cis=[]
    for trait in traitChr:
        cis+=[ail_cis[ail_cis['chrom']==trait].sort_values(by='raw p-value')[['gene_name','cis-eQTL snp']].iloc[0:B]]
    
    futures=[]
    manhattanPlotsHelp(traitChr[0],trans[0],cis[0],parms,B)
    with ProcessPoolExecutor(smallCpu) as executor: 
        for j in range(len(traitChr)):
            trait=traitChr[j]
            futures.append(executor.submit(manhattanPlotsHelp,trait,trans[j],cis[j],parms,B))
            
        wait(futures,return_when=ALL_COMPLETED)
    
    ############ trans #########################3
    fig,axs=plt.subplots(len(traitChr),B,dpi=50)

    fig.set_figwidth(20*B,forward=True)
    fig.set_figheight(190,forward=True)
    
    for j in range(len(traitChr)):
        trait=traitChr[j]

        with open(scratchDir+'ma-'+trait+'-trans.pickle', 'rb') as handle:
            ret = pickle.load(handle)

        transLoc=ret['loc']
        P=ret['P']
                
        for i in range(P.shape[1]):            
            traitName=trans[j]['target_gene'].iloc[i]
            snp='chr'+str(int(trans[j]['eqtl_chrom'].iloc[i]))
            Mbp=trans[j]['eqtl_pos_bp'].iloc[i]
            
            axs[j,i].plot(snpData['Mbp'][snpData['chr']==snp],-np.log10(P[snpData['chr']==snp,i]),'ko')  

            axs[j,i].set_xlabel(trait+' - '+traitName)
            axs[j,i].set_ylim([0,4])
            axs[j,i].set_xlim([0,np.max(np.append(snpData['Mbp'][snpData['chr']==snp].values.flatten(),[Mbp]))])

            axs[j,i].axvline(x=Mbp,color='g',linewidth=4,label='eqtl')
            
            axs[j,i].legend()
                        
    fig.savefig(plotsDir+'manhattan-trans.png',bbox_inches='tight')
    plt.close('all')

    ############ Cis #########################3
    
    fig,axs=plt.subplots(len(traitChr),B,dpi=50)

    fig.set_figwidth(20*B,forward=True)
    fig.set_figheight(190,forward=True)

    for j in range(len(snpChr)):
        trait=snpChr[j] # snpChr since traitChr has extra chr

        if not os.path.isfile(scratchDir+'ma-'+trait+'-cis.pickle'):
            continue

        with open(scratchDir+'ma-'+trait+'-cis.pickle', 'rb') as handle:
            ret = pickle.load(handle)

        cisLoc=ret['loc']
        P=ret['P']

        for i in range(P.shape[1]):            
            traitName=cis[j]['gene_name'].iloc[i]
            traitMbp=float(traitData['Mbp'][traitData['chr']==trait].iloc[cisLoc[i]])
            eqtlMbp=float(re.sub(r'^chr[0-9]+\.','',cis[j]['cis-eQTL snp'].iloc[i]))
            
            axs[j,i].plot(snpData['Mbp'][snpData['chr']==trait],-np.log10(P[:,i]),'ko')  

            axs[j,i].set_xlabel(trait+' - '+traitName)
            axs[j,i].set_ylim([0,4])
            axs[j,i].set_xlim([0,np.max(np.append(snpData['Mbp'][snpData['chr']==trait].values.flatten(),[traitMbp,eqtlMbp]))])

            axs[j,i].axvline(x=traitMbp,color='r',linewidth=4,label='trait')
            axs[j,i].axvline(x=eqtlMbp,color='g',linewidth=4,label='eqtl')
            
            axs[j,i].legend()
                        
    fig.savefig(plotsDir+'manhattan-cis.png',bbox_inches='tight')
    plt.close('all')
    
def manhattanPlotsHelp(trait,trans,cis,parms,B): 
    scratchDir=parms['scratchDir']
    snpChr=parms['snpChr']
    
    traitData=pd.read_csv(scratchDir+'traitData.csv')
    traitInd=traitData['trait'][traitData['chr']==trait].to_frame()
    traitInd.insert(1,'ind',np.arange(sum(traitData['chr']==trait)))
    
    #if os.path.isfile(scratchDir+'ma-'+trait+'-trans.pickle'):
    #    return()
        
    if trait in snpChr:
        cisLoc=traitInd.merge(cis,left_on='trait',right_on='gene_name')['ind'].values.flatten()
    transLoc=traitInd.merge(trans,left_on='trait',right_on='target_gene')['ind'].values.flatten()
    
    transP=[]
    for snp in snpChr: 
        if snp==trait:
            cisP=np.loadtxt(scratchDir+'p-'+snp+'-'+trait+'.csv',delimiter=',')[:,cisLoc]
        transP+=[np.loadtxt(scratchDir+'p-'+snp+'-'+trait+'.csv',delimiter=',')[:,transLoc]]
    
    #pdb.set_trace()
    transP=np.concatenate(transP,axis=0)
           
    print('writing pickle for ',trait,flush=True)

    if trait in snpChr:
        with open(scratchDir+'ma-'+trait+'-cis.pickle', 'wb') as handle:
            pickle.dump({'loc':cisLoc, 'P':cisP},handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open(scratchDir+'ma-'+trait+'-trans.pickle', 'wb') as handle:
        pickle.dump({'loc':transLoc,'P':transP},handle,protocol=pickle.HIGHEST_PROTOCOL)

    return()

