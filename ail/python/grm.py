import pandas as pd
import subprocess
import pdb
import os

def grm(ch,snps,files):
    gemma=files['gemma']
    genoFile='geno-'+str(ch)+'.txt'
    
    snps[snpChr!=ch].to_csv(genoFile,sep=' ',index=False,header=False)

    # generate loco
    subprocess.run([gemma,'-g',genoFile,'-p','pheno.txt','-gk','2','-o','grm-'+str(ch)])

    # move grm to scratch
    os.rename(scratchDir+'output/grm-'+str(ch)+'.sXX.txt',scratchDir+'grm-'+str(ch)+'.sXX.txt')
    
