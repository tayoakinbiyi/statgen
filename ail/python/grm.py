import pandas as pd
import numpy as np
import os
import pdb
import subprocess

def grm(snp,snpChr,snps,files):
    snps[snpChr!=snp].to_csv('geno-grm'+snp+'.txt',sep='\t',index=False,header=False)

    # generate loco
    subprocess.run([files['gemma'],'-g','geno-grm'+snp+'.txt','-p','dummy.txt','-gk','1','-o','grm-'+snp])

    # move grm to scratch
    os.rename('output/grm-'+snp+'.cXX.txt','grm-'+snp+'.txt')

    os.remove('geno-grm'+snp+'.txt')
    os.remove('output/geno-grm'+snp+'.log.txt')