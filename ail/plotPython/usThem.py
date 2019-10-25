import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from ail.opPython.DB import *

def usThem(parms):
    local=parms['local']
    name=parms['name']
    traitChr=parms['traitChr']
    smallNumCores=parms['smallNumCores']
    
    DBCreateFolder(name,'usThemPlots',parms)

    futures=[]
    #usThemHelp('chr1',parms)
    with ProcessPoolExecutor(smallNumCores) as executor: 
        for trait in traitChr:
            #usThemHelp('chr2',parms)
            if DBIsFile(name+'holds',trait,parms):
                continue
                
            DBWrite(np.array([]),name+'holds/'+trait,parms,True)
    
            futures.append(executor.submit(usThemHelp,trait,parms))
            
        wait(futures,return_when=ALL_COMPLETED)
    
    fig,axs=plt.subplots(1,1,dpi=50)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    df=[]
    for trait in traitChr:
        df+=[DBRead(name+'usThem/'+trait,parms,toPickle=True)]
       
    df=pd.DataFrame(np.concatenate(df,axis=0),columns=['eqtl_pvalue','pval'])
    
    q=10*np.arange(0,11)
    val=(df['eqtl_pvalue']-df['pval']).abs().values.flatten()
    pct=((df['eqtl_pvalue']-df['pval']).abs()/df['eqtl_pvalue']).values.flatten()
    pd.DataFrame({'Percentile':q,'Difference':np.percentile(val,q),'% error':np.round(100*np.percentile(pct,q),3)}).to_csv(
        local+name+'plots/usThemDiff.csv',index=False)
    DBUpload(name+'plots/usThemDiff.csv',parms,toPickle=False)
    
    df['eqtl_pvalue']=-np.log10(df['eqtl_pvalue'])
    df['pval']=-np.log10(df['pval'])
    
    df.plot.scatter(x='eqtl_pvalue',y='pval',ax=axs)
    
    maxP=max(df['pval'].max(),df['eqtl_pvalue'].max())

    axs.set_ylabel('ours')
    axs.set_xlabel('theirs')
    axs.set_xlim([0,maxP])
    axs.set_ylim([0,maxP])
    
    axs.plot(axs.get_xlim(), axs.get_ylim(), ls="--", c=".3") 
    axs.set_title('us vs them')
        
    if not os.path.exists(local+name+'usThemPlots'):
        os.mkdir(local+name+'usThemPlots')
        
    fig.savefig(local+name+'usThemPlots/usThem.png',bbox_inches='tight')
    plt.close('all')
    
    DBUpload(name+'usThemPlots/usThem.png',parms,toPickle=False)
    
def usThemHelp(trait,parms):
    local=parms['local']
    name=parms['name']
    wald=parms['wald']
    snpChr=parms['snpChr']
    
    snpChr=[x for x in snpChr if x!=trait]
    
    snpData=DBLocalRead(name+'process/snpData',parms)
    snpData=snpData[snpData['chr']!=trait]

    traitData=DBLocalRead(name+'process/traitData',parms)
    traitData=traitData[traitData['chr']==trait]

    ail_paper=pd.read_csv(local+'data/ail_paper-Trans.csv',header=0)
    ail_paper=ail_paper[(ail_paper['eqtl_tissue']=='hip')&(ail_paper['target_gene_chrom']==int(trait[3:]))]
    ail_paper=ail_paper[['eqtl_pos_bp','eqtl_chrom','eqtl_pvalue','target_gene']].reset_index(drop=True)
    
    ail_paper=ail_paper.merge(pd.DataFrame({'target_gene':traitData['trait'],'loc':np.arange(len(traitData))}),on='target_gene')
    traitList=ail_paper['loc'].values.flatten()
    
    pval={}
    for snp in snpChr:
        snpChrom=int(snp[3:])
        print('loading pvals from snp '+snp+' trait '+trait)
        pval[snpChrom]=DBRead(name+'score/p-'+snp+'-'+trait,parms,toPickle=True)[:,traitList]
        if wald:
            pval[snpChrom]=2*norm.sf(np.abs(pval[snpChrom]))
    
    ans=[]
    for ind,eqtl in ail_paper.iterrows():
        if not ('chr'+str(int(eqtl['eqtl_chrom'])) in snpChr):
            continue
        t_snpData=snpData[snpData['chr']=='chr'+str(int(eqtl['eqtl_chrom']))]
        ans+=[[eqtl['eqtl_pvalue'],np.min(pval[int(eqtl['eqtl_chrom'])][(t_snpData['Mbp']<eqtl['eqtl_pos_bp']+1e6)&
            (t_snpData['Mbp']>eqtl['eqtl_pos_bp']-1e6),ind].flatten())]]
    
    ans=np.array(ans)
    DBWrite(ans,name+'usThemPlots/'+trait,parms,True)
    