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
        sys.stderr.write(snp+'-'+trait+'-'+str(k))

        subprocess.run([gemma,'-g','geno-'+str(snp)+'.txt','-p','pheno-'+trait+'.txt','-c','PC.csv','-lmm','2','-o',
            'pvals-'+snp+'-'+trait,'-k','grm-'+snp+'.sXX.txt','-maf','0.05','-r2','0.99','-n',str(k+1)])
        
        lmm[traitChr.trait.iloc[k]]=pd.read_csv('output/pvals-'+snp+'-'+trait+'.assoc.txt',sep='\t')['p_score'].values.flatten()
      
    lmm.to_csv('pvals-final-'+snp+'-'+trait+'.csv')
    