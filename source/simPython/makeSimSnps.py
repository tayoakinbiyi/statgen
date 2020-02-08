import numpy as np
import pandas as pd
import random
import subprocess

def makeSimSnps(parms): 
    snpSize=parms['snpSize']
    maxSnpGen=parms['maxSnpGen']
    local=parms['local']
    numSubjects=parms['numSubjects']
    
    numSnps=np.sum(snpSize)
    subprocess.call(['cp',local+'ext/sampleIds.txt','geneDrop/sampleIds.txt'])
    pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
        'geneDrop/parms.txt',index=False,header=None)

    cols=[]
    t_numSnps=0
    while t_numSnps<numSnps:
        newAdd=min(maxSnpGen,t_numSnps-numSnps)
        pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
            'geneDrop/map.txt',sep='\t',index=False)

        rows=[]
        t_numSubjects=0
        while t_numSubjects<numSubjects:
            cmd=[local+'ext/gdrop','-p','geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o','geneDrop/geneDrop']
            subprocess.run(cmd)

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