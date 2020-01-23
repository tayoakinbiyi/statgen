import numpy as np
import pandas as pd
import random
import subprocess

def makeSimSnps(parms): 
    SnpSize=parms['SnpSize']
    maxSnpGen=parms['maxSnpGen']
    local=parms['local']

    snps=[]
    numSnps=0
    size=np.sum(SnpSize)
    while numSnps<size:
        newAdd=min(maxSnpGen,size-numSnps)
        pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
            'geneDrop/map.txt',sep='\t',index=False)
        cmd=[local+'ext/gdrop','-p','geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o','geneDrop/geneDrop']
        subprocess.run(cmd)

        val=pd.read_csv('geneDrop/geneDrop.geno_true',header=0,sep='\t').iloc[:,4:]

        valMAF=np.concatenate([(col.str.split(' ',expand=True).astype(int)>2).sum(axis=1).values.reshape(-1,1) for ind,
            col in val.iteritems()],axis=1)
        val=pd.concat([col.str.split(' ',expand=True).replace({'1':'A','2':'A','3':'G','4':'G'}).apply(lambda x:' '.join(x),axis=1)
            for ind,col in val.iteritems()],axis=1)

        maf=np.mean(valMAF,axis=1)/2
        maf=np.minimum(maf,1-maf)
        val=val.loc[maf>.1,:]

        numSnps+=val.shape[0]
        print('removed '+str(newAdd-val.shape[0])+' snps',flush=True)

        snps+=[val]
    
    return(pd.concat(snps,axis=0).T.reset_index(drop=True))