import numpy as np
import pandas as pd
import random
import subprocess
import pdb
from ELL.util import *
from multiprocessing import cpu_count

def makePedigreeSnps(numSubjects,miceRange,numSnps,maxGen=5000,numCores=cpu_count(),maxSnpGen=5000):
    subprocess.call(['rm','-rf','geneDrop'])
    subprocess.call(['mkdir','-p','geneDrop'])
    b_snps=bufCreate('snps',[numSubjects,numSnps])   

    np.savetxt('geneDrop/sampleIds.txt',np.loadtxt('../ext/sampleIds.txt',delimiter='\t',dtype=str)[miceRange],
        delimiter='\t',fmt='%s')
    
    seeds=np.random.randint(int(1e6),size=numCores)
    uniqueMice=len(miceRange)

    pids=[]
    for core in range(numCores):
        snpRange=np.arange(core*int(np.ceil(numSnps/numCores)),min(numSnps,(core+1)*int(np.ceil(numSnps/numCores))))
        if len(snpRange)==0:
            continue
        
        pids+=[remote(makePedigreeSnpsHelp,core,b_snps,snpRange,uniqueMice,seeds[core],maxSnpGen)]

    for pid in pids:
        os.waitpid(0, 0)
    
    snps=bufClose(b_snps).astype(int)
    
    return(snps)

def makePedigreeSnpsHelp(core,b_snps,snpRange,uniqueMice,seed,maxSnpGen):
    numSubjects=b_snps[0].shape[0]
    numSnps=len(snpRange)
    
    numBlocks=int(np.ceil(numSubjects/uniqueMice))
    numReps=numBlocks*numSnps
    
    np.random.seed(seed)

    pd.DataFrame({'parms':['../ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map-'+str(core)+
        '.txt',0,0]}).to_csv('geneDrop/parms-'+str(core)+'.txt',index=False,header=None)
    
    tab=str.maketrans('1234 ','0011.')
    
    snps=[]
    t_numReps=0
    count=1
    while t_numReps<numReps:
        newAdd=min(maxSnpGen,numReps-t_numReps)
        pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
            'geneDrop/map-'+str(core)+'.txt',sep='\t',index=False)

        seed=np.random.randint(1,1e6)
        print(t_numReps,seed)
        cmd=['../ext/gdrop','-p','geneDrop/parms-'+str(core)+'.txt','-s',str(seed),'-o','geneDrop/geneDrop-'+str(core)]
        subprocess.run(cmd)
        
        new=np.loadtxt('geneDrop/geneDrop-'+str(core)+'.geno_true',delimiter='\t',dtype=str)[:,4:] # snps x mice
        snps+=[new.reshape(len(new),-1)]            
        t_numReps+=snps[-1].shape[0]
        count+=1

    snps=np.concatenate(snps,axis=0)
    
    snps=np.ceil(np.char.translate(snps,tab).astype(float)).astype(int)
    b_snps[0][:,snpRange]=np.hstack(np.vsplit(snps,numBlocks))[:numSnps,:numSubjects].T
    b_snps[1].flush()
    
    return()
