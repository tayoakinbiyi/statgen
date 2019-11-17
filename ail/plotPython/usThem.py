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
    
    DBCreateFolder('usThem',parms)

    for trait in traitChr:
        if DBIsFile('holds/'+trait,parms):
            continue

        DBWrite(np.array([]),'holds/'+trait,parms)

        usThemHelp(trait,parms)
    
    fig,axs=plt.subplots(1,1,dpi=50)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    df=[]
    for trait in traitChr:
        df+=[DBRead('usThem/'+trait,parms)]
       
    df=pd.DataFrame(np.concatenate(df,axis=0),columns=['eqtl_pvalue','pval'])
    
    q=10*np.arange(0,11)
    val=(df['eqtl_pvalue']-df['pval']).abs().values.flatten()
    pct=((df['eqtl_pvalue']-df['pval']).abs()/df['eqtl_pvalue']).values.flatten()
    pd.DataFrame({'Percentile':q,'Difference':np.percentile(val,q),'% error':np.round(100*np.percentile(pct,q),3)}).to_csv(
        'usThem/usThemDiff.csv',index=False)
    
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
                
    fig.savefig('usThem/usThem.png',bbox_inches='tight')
    plt.close('all')
    
    return()
        
def usThemHelp(trait,parms):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']
    
    snpChr=[x for x in snpChr if x!=trait]
    
    snpData=DBRead('ped/snpData',parms)
    snpData=snpData[snpData['chr']!=trait]

    traitData=DBRead('ped/traitData',parms)
    traitData=traitData[traitData['chr']==trait]

    ail_paper=pd.read_csv(local+'data/ail_paper-Trans.csv',header=0)
    ail_paper=ail_paper[(ail_paper['eqtl_tissue']=='hip')&(ail_paper['target_gene_chrom']==int(trait))]
    ail_paper=ail_paper[['eqtl_pos_bp','eqtl_chrom','eqtl_pvalue','target_gene']].reset_index(drop=True)
    ail_paper.loc[:,'eqtl_chrom']=ail_paper.loc[:,'eqtl_chrom'].astype(int).astype(str)
    
    ail_paper=ail_paper.merge(pd.DataFrame({'target_gene':traitData['trait'],'loc':np.arange(len(traitData))}),on='target_gene')
    traitList=ail_paper['loc'].values.flatten()
    
    pval={}
    for snp in snpChr:
        print('loading pvals from snp '+snp+' trait '+trait)
        pval[snp]=2*norm.sf(np.abs(DBRead('score/'+snp+'-'+trait,parms)))
    
    ans=[]
    for ind,eqtl in ail_paper.iterrows():
        if not (eqtl['eqtl_chrom'] in snpChr):
            continue
            
        t_snpData=snpData[snpData['chr']==eqtl['eqtl_chrom']]
        ans+=[[eqtl['eqtl_pvalue'],np.min(pval[eqtl['eqtl_chrom']][(t_snpData['Mbp']<eqtl['eqtl_pos_bp']+1e6)&
            (t_snpData['Mbp']>eqtl['eqtl_pos_bp']-1e6),ind].flatten())]]
    
    ans=np.array(ans)
    DBWrite(ans,'usThem/'+trait,parms)
    