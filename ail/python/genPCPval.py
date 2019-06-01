import pandas as pd
import subprocess
import pdb
import os
import sys

def genPCPval(snp,snpId,numPCs,files):
    gemma=files['gemma']
    
    if os.path.isfile('pvals-final-'+snp+'-'+trait+'.txt'):
        pass
        #return()

    PCs=np.empty([len(snpId),numPCs])

    for k in range(numPCs):
        subprocess.run([gemma,'-g','geno-'+str(snp)+'.csv','-p','PCAll.txt','-lmm','2','-o',
            'pvals-'+snp,'-k','grm-'+snp+'.sXX.csv','-maf','0.05','-r2','0.99','-n',str(k+1)])
        
        PCs[:,k]=pd.read_csv('output/pvals-'+snp+'.assoc.txt',sep='\t')['p_score'].values.flatten()
      
    np.savetxt('pvals-PC-'+snp+'.txt',PCs,delimiter='\t')
    