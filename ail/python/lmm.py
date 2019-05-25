import pandas as pd
import subprocess
import pdb
import os

def lmm(ch,snpId,traitChr,files):
    print('here '+ch)
    gemma=files['gemma']

    lmm=pd.DataFrame(index=pd.MultiIndex.from_product([[ch],snpId],names=['chr','Mbp']),
                          columns=pd.MultiIndex.from_tuples(traitChr.values.tolist(),names=['trait','chr','Mbp']))

    for ph in range(traitChr.shape[0]):
        print(ph,ch)

        subprocess.run([gemma,'-g','geno-'+str(ch)+'.txt','-p','pheno.txt','-c','PC.txt','-lmm','3','-o','pvals-'+str(ch),'-k',
                        'grm-'+str(ch)+'.sXX.txt','-maf','0.05','-r2','0.99','-n',str(ph+1)])
        
        lmm[traitChr.trait.iloc[ph]]=pd.read_csv('output/pvals-'+str(ch)+'.assoc.txt',sep='\t')['p_score'].values.flatten()
      
    lmm.to_csv('pvals-final-'+ch+'.txt')
    
    return()