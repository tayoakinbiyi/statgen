import pandas as pd
import numpy as np
import subprocess
import pdb
import os
import sys

def genPCPval(snp,snpId,numPCs,files):
    gemma=files['gemma']
    
    if os.path.isfile('pvals-PC-'+snp+'.txt'):
        return()
    
    PCs=np.empty([len(snpId),numPCs])

    for k in range(numPCs):
        subprocess.run([gemma,'-g','geno-'+str(snp)+'.txt','-p','PCAll.txt','-lmm','2','-o','pvals-'+snp,'-k','grm-'+snp+'.txt','-maf',
            '0.05','-r2','0.99','-n',str(k+2)])
        
        os.remove('output/pvals-'+snp+'.log.txt')
        
        PCs[:,k]=pd.read_csv('output/pvals-'+snp+'.assoc.txt',sep='\t')['p_lrt'].values.flatten()
    
        os.remove('output/pvals-'+snp+'.assoc.txt')
    
    sys.stderr.write(snp) 
    np.savetxt('pvals-PC-'+snp+'.txt',PCs,delimiter='\t')
    