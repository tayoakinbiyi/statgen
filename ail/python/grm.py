import pandas as pd
import numpy as np
import os
import pdb
import subprocess

def grm(snp,snpChr,snps,files):
    snps[snpChr!=snp].to_csv('geno-'+snp+'.txt',sep=' ',index=False,header=False)

    # generate loco
    subprocess.run([files['gemma'],'-g','geno-'+snp+'.txt','-p','dummy.txt','-gk','2','-o','grm-'+snp])

    # move grm to scratch
    os.rename('output/grm-'+snp+'.sXX.txt','grm-'+snp+'.sXX.txt')

    # write chr gene file
    snps[snpChr==snp].to_csv('geno-'+snp+'.txt',sep=' ',index=False,header=False)
