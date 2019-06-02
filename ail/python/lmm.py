import pandas as pd
import subprocess
import pdb
import os
import sys

def lmm(snp,trait,snpId,traitChr,files):
    gemma=files['gemma']
    
    if os.path.isfile('pvals-final-'+snp+'-'+trait+'.txt'):
        return()

    lmm=pd.DataFrame(index=pd.MultiIndex.from_product([[snp],snpId],names=['chr','Mbp']),
        columns=pd.MultiIndex.from_tuples(traitChr.values.tolist(),names=['trait','chr','Mbp']))

    for k in range(traitChr.shape[0]):
        subprocess.run([gemma,'-g','geno-'+str(snp)+'.txt','-p','pheno-'+trait+'.txt','-c','preds.txt','-lmm','2','-o',
            'pvals-'+snp+'-'+trait,'-k','grm-'+snp+'.txt','-maf','0.05','-r2','0.99','-n',str(k+1)])
        
        os.remove('output/pvals-'+snp+'-'+trait+'.log.txt')
        
        lmm[traitChr.trait.iloc[k]]=pd.read_csv('output/pvals-'+snp+'-'+trait+'.assoc.txt',sep='\t')['p_lrt'].values.flatten()

        os.remove('output/pvals-'+snp+'-'+trait+'.assoc.txt')
      
    sys.stderr.write(snp+'-'+trait)
    lmm.to_csv('pvals-final-'+snp+'-'+trait+'.csv')
    