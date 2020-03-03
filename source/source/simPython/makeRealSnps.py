import numpy as np
import pandas as pd
import random
import subprocess

def makeRealSnps(parms): 
    snpSize=parms['snpSize']
    local=parms['local']
    numSubjects=parms['numSubjects']
    
    numSnps=np.sum(snpSize)
    snps=np.loadtxt(local+'data/snps.txt',delimiter='\t')[:,2:].T

    cols=[]
    t_numSnps=0
    while t_numSnps<numSnps:
        rows=[]
        t_numSubjects=0
        while t_numSubjects<numSubjects:
            snpList+=[snps[np.arange(min(208,numSubjects-t_num))][:,random.sample(range(snps.shape[1]),numSnps)]]        
            row+=[np.loadtxt('geneDrop/geneDrop.geno_true',delimiter='\t')[(numSnps-t_numSnps),4:4+(numSubjects-t_numSubjects)].T]

            t_numSubjects+=row[-1].shape[0]
            
        rows=np.concatenate(rows,axis=0)
        
        maf=np.mean(np.apply_over_axes(lambda string: np.sum(np.array(string.split(' ')).isin(['3','4'])),rows,[0,1]),axis=0)/2
        maf=np.minimum(maf,1-maf)
        rows=rows[:,maf>.1]
        rows=np.char.translate(rows,{'1':'A','2':'A','3':'G','4':'G'})
        
        cols+=[rows]
        t_numSnps+=cols[-1].shape[1]

    snps=np.concatenate(cols,axis=1)
    
    return(snps)