import numpy as np
import pandas as pd
import random
import subprocess
import pdb

def makeSimSnps(parms): 
    snpSize=parms['snpSize']
    maxSnpGen=parms['maxSnpGen']
    local=parms['local']
    numSubjects=parms['numSubjects']
        
    numSnps=np.sum(snpSize)
    subprocess.call(['cp',local+'ext/sampleIds.txt','geneDrop/sampleIds.txt'])
    pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
        'geneDrop/parms.txt',index=False,header=None)
    
    tab=str.maketrans('1234 ','0011.')

    rows=[]
    t_numSnps=0
    while t_numSnps<numSnps:
        newAdd=min(maxSnpGen,numSnps-t_numSnps)
        pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
            'geneDrop/map.txt',sep='\t',index=False)

        row=[]
        t_numSubjects=0
        while t_numSubjects<numSubjects:
            cmd=[local+'ext/gdrop','-p','geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o','geneDrop/geneDrop']
            subprocess.run(cmd)
            row+=[np.loadtxt('geneDrop/geneDrop.geno_true',delimiter='\t',dtype=str)[:,4:4+(numSubjects-t_numSubjects)]]

            t_numSubjects+=row[-1].shape[1]
            
        row=np.concatenate(row,axis=1)
        row=np.ceil(np.char.translate(row,tab).astype(float))
        af=np.mean(row,axis=1)/2
        maf=np.minimum(af,1-af)
        row=row[maf>.1,:]
        
        rows+=[row]
        t_numSnps+=rows[-1].shape[0]

    bimBamFmt=np.concatenate(rows,axis=0)
    
    return(bimBamFmt)