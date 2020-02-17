import pandas as pd
import subprocess
import numpy as np
import json

from opPython.DB import *

def writeSnps(snps,snpData,parms):
    snpChr=parms['snpChr']
    local=parms['local']
    
    mouseIds=np.arange(len(snps))
    
    fam=pd.DataFrame({'Family ID':mouseIds,'Individual ID':0,'Paternal ID':mouseIds,'Maternal ID':0,'Sex':1,'Phenotype':1})

    for snp in snpChr:
        pd.concat([fam,snps.loc[:,snpData['chr'].values==snp]],axis=1).to_csv(
            'ped/snp-'+str(snp)+'.ped',header=False,index=False,sep='\t')   
        snpData[snpData['chr'].values==snp].to_csv('ped/snp-'+str(snp)+'.map',header=False,index=False,sep='\t')  
        pd.DataFrame({'id':range(sum(snpData['chr']==snp)),'allele':'A'}).to_csv('ped/ref-'+str(snp),sep='\t',
            index=False,header=False)
        subprocess.call([local+'ext/plink','--reference-allele','ped/ref-'+str(snp),'--file','ped/snp-'+str(snp),'--out',
            'ped/snp-'+str(snp),'--make-bed','--noweb'])
    
    pd.concat([fam,snps],axis=1).to_csv('ped/snp.ped',header=False,index=False,sep='\t')   
    snpData.to_csv('ped/snp.map',header=False,index=False,sep='\t')  
    pd.DataFrame({'id':range(len(snpData)),'allele':'A'}).to_csv('ped/ref',sep='\t',index=False,header=False)
    subprocess.call([local+'ext/plink','--reference-allele','ped/ref','--file','ped/snp','--out','ped/snp','--make-bed',
        '--noweb'])
    
    maf=np.array([col.str.split(' ',expand=True).apply(lambda df: df.value_counts(),axis=0).sum(axis=1).min() 
                  for ind,col in snps.iteritems()])/(2*len(snps))
    
    snpMinor=np.array([col.str.split(' ',expand=True).apply(lambda df: df.value_counts(),axis=0).sum(axis=1).idxmin() 
                  for ind,col in snps.iteritems()])

    snpData.insert(snpData.shape[1],'maf',maf)
    snpData.insert(snpData.shape[1],'minor',snpMinor)
    snpData.to_csv('ped/snpData',index=False,sep='\t')

    DBLog('number of snps by chromosome \n'+json.dumps(snpData.groupby('chr')['ID'].count().to_dict(),indent=3))                
    return()