import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED

home='/project/abney/'

files={
    'dataDir':home+'ail/data/',
    'scratchDir':home+'ail/scratch/',
    'gemma':home+'ail/gemma'
}

#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

def chFix(snp,scratchDir):
    pval=[]

    for k in range(10):
        pval+=[pd.read_csv(scratchDir+'pvals-final-'+snp+'-'+str(k)+'.txt',index_col=[0,1],header=[0,1,2])]
    
    pval=pd.concat(pval,axis=1)
    
    chrList=pval.columns.levels[1]
    for trait in chrList:
        pval.loc[:,(slice(None),snp,slice(None))].to_csv(scratchDir+'pvals-final-'+snp+'-'+trait+'.txt')
       
futures=[]
with ProcessPoolExecutor() as executor: 
    for snp in ['chr'+str(x) for x in range(1,20)]:
        futures.append(executor.submit(chFix,snp,scratchDir))
        
wait(futures,return_when=ALL_COMPLETED)
